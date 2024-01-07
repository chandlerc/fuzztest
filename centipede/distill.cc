// Copyright 2023 The Centipede Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "./centipede/distill.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "./centipede/blob_file.h"
#include "./centipede/defs.h"
#include "./centipede/environment.h"
#include "./centipede/feature.h"
#include "./centipede/feature_set.h"
#include "./centipede/logging.h"
#include "./centipede/rusage_profiler.h"
#include "./centipede/shard_reader.h"
#include "./centipede/thread_pool.h"
#include "./centipede/util.h"
#include "./centipede/workdir.h"

namespace centipede {

namespace {

struct CorpusElt {
  ByteArray input;
  FeatureVec features;

  ByteArray PackedFeatures() const {
    return PackFeaturesAndHash(input, features);
  }
};

using CorpusEltVec = std::vector<CorpusElt>;

// The maximum number of threads reading input shards concurrently. This is
// mainly to prevent I/O congestion.
// TODO(ussuri): Bump up significantly when RSS-gated mutexing is in.
inline constexpr size_t kMaxReadingThreads = 1;
// The maximum number of threads writing shards concurrently. These in turn
// launch up to `kMaxReadingThreads` reading threads.
inline constexpr size_t kMaxWritingThreads = 10;
// A global cap on the total number of threads, both writing and reading. Unlike
// the other two limits, this one is purely to prevent too many threads in the
// process.
inline constexpr size_t kMaxTotalThreads = 1000;
static_assert(kMaxReadingThreads * kMaxWritingThreads <= kMaxTotalThreads);

std::string LogPrefix(const Environment &env) {
  return absl::StrCat("DISTILL[S.", env.my_shard_index, "]: ");
}

// TODO(ussuri): Move the reader/writer classes to shard_reader.cc, rename it
//  to corpus_io.cc, and reuse the new APIs where useful in the code base.

// A helper class for reading input corpus shards. Thread-safe.
class InputCorpusShardReader {
 public:
  InputCorpusShardReader(const Environment &env) : env_{env} {}

  // Reads and returns a single shard's elements. Thread-safe.
  CorpusEltVec ReadShard(size_t shard_idx) {
    const WorkDir wd{env_};
    const auto corpus_path = wd.CorpusFiles().ShardPath(shard_idx);
    const auto features_path = wd.FeaturesFiles().ShardPath(shard_idx);
    VLOG(1) << LogPrefix(env_) << "reading input shard " << shard_idx << ":\n"
            << VV(corpus_path) << "\n"
            << VV(features_path);
    CorpusEltVec elts;
    // Read elements from the current shard.
    centipede::ReadShard(  //
        corpus_path, features_path,
        [&elts](const ByteArray &input, FeatureVec &features) {
          elts.emplace_back(input, std::move(features));
        });
    ++num_read_shards_;
    return elts;
  }

  size_t num_read_shards() const { return num_read_shards_; }

 private:
  Environment env_;
  std::atomic<size_t> num_read_shards_ = 0;
};

// A helper class for writing corpus shards. Thread-safe by virtue of enforcing
// exclusive locking in the function annotations.
class CorpusShardWriter {
 public:
  CorpusShardWriter(const Environment &env, std::string_view mode)
      : env_{env},
        corpus_writer_{DefaultBlobFileWriterFactory()},
        feature_writer_{DefaultBlobFileWriterFactory()} {
    const WorkDir wd{env};
    corpus_path_ = wd.DistilledCorpusFiles().MyShardPath();
    features_path_ = wd.DistilledFeaturesFiles().MyShardPath();
    CHECK_OK(corpus_writer_->Open(corpus_path_, mode));
    CHECK_OK(feature_writer_->Open(features_path_, mode));
  }

  virtual ~CorpusShardWriter() = default;

  void WriteElt(CorpusElt elt) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    ++num_total_elts_;
    if (PreprocessElt(elt) == EltDisposition::kWrite) {
      // Append to the distilled corpus and features files.
      CHECK_OK(corpus_writer_->Write(elt.input));
      CHECK_OK(feature_writer_->Write(elt.PackedFeatures()));
      ++num_written_elts_;
    }
  }

  void WriteBatch(CorpusEltVec elts) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    VLOG(1) << LogPrefix(env_) << "writing " << elts.size()
            << " elements to output shard:\n"
            << VV(corpus_path_) << "\n"
            << VV(features_path_);
    for (auto &elt : elts) {
      WriteElt(std::move(elt));
    }
    ++num_written_batches_;
  }

  size_t num_total_elts() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return num_total_elts_;
  }
  size_t num_written_elts() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return num_written_elts_;
  }
  size_t num_written_batches() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return num_written_batches_;
  }

  absl::Mutex &Mutex() ABSL_LOCK_RETURNED(mu_) { return mu_; }

 protected:
  [[nodiscard]] enum class EltDisposition { kWrite, kIgnore };

  // A behavior customization point: a derived class gets an opportunity to
  // analyze and/or preprocess `elt` before it is written. In particular, the
  // derived class can choose to skip writing entirely by returning `kIgnore`.
  virtual EltDisposition PreprocessElt(CorpusElt &elt) {
    return EltDisposition::kWrite;
  }

 private:
  Environment env_;
  std::string corpus_path_;
  std::string features_path_;

  absl::Mutex mu_;

  std::unique_ptr<BlobFileWriter> corpus_writer_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<BlobFileWriter> feature_writer_ ABSL_GUARDED_BY(mu_);
  size_t num_total_elts_ ABSL_GUARDED_BY(mu_) = 0;
  size_t num_written_elts_ ABSL_GUARDED_BY(mu_) = 0;
  size_t num_written_batches_ ABSL_GUARDED_BY(mu_) = 0;
};

// A helper class for writing distilled corpus shards. NOT thread-safe because
// all writes go to a single file.
class DistilledCorpusShardWriter : public CorpusShardWriter {
 public:
  DistilledCorpusShardWriter(const Environment &env, std::string_view mode)
      : CorpusShardWriter{env, mode},
        feature_set_(/*frequency_threshold=*/1, env.MakeDomainDiscardMask()) {}

  ~DistilledCorpusShardWriter() override = default;

  const FeatureSet &feature_set() { return feature_set_; }

 protected:
  EltDisposition PreprocessElt(CorpusElt &elt) override {
    feature_set_.PruneDiscardedDomains(elt.features);
    if (!feature_set_.HasUnseenFeatures(elt.features))
      return EltDisposition::kIgnore;
    feature_set_.IncrementFrequencies(elt.features);
    return EltDisposition::kWrite;
  }

 private:
  FeatureSet feature_set_;
};

}  // namespace

void DistillTask(const Environment &env,
                 const std::vector<size_t> &shard_indices) {
  // Read and write the shards in parallel, but gate reading of each on the
  // availability of free RAM to keep the peak RAM usage under control.
  const size_t num_shards = shard_indices.size();
  InputCorpusShardReader reader{env};
  // NOTE: Always overwrite corpus and features files, never append.
  DistilledCorpusShardWriter writer{env, "w"};

  {
    ThreadPool threads{kMaxReadingThreads};
    for (size_t shard_idx : shard_indices) {
      threads.Schedule([shard_idx, &reader, &writer, &env, num_shards] {
        CorpusEltVec shard_elts = reader.ReadShard(shard_idx);
        // Reverse the order of elements. The intuition is as follows:
        // * If the shard is the result of fuzzing with Centipede, the inputs
        //   that are closer to the end are more interesting, so we start there.
        // * If the shard resulted from somethening else, the reverse order is
        //   not any better or worse than any other order.
        std::reverse(shard_elts.begin(), shard_elts.end());
        {
          absl::WriterMutexLock lock(&writer.Mutex());
          writer.WriteBatch(std::move(shard_elts));
          LOG(INFO) << LogPrefix(env) << writer.feature_set()
                    << " src_shards: " << writer.num_written_batches() << "/"
                    << num_shards << " src_elts: " << writer.num_total_elts()
                    << " dst_elts: " << writer.num_written_elts();
        }
      });
    }
  }  // The reading threads join here.
}

int Distill(const Environment &env) {
  RPROF_THIS_FUNCTION_WITH_TIMELAPSE(                                 //
      /*enable=*/VLOG_IS_ON(1),                                       //
      /*timelapse_interval=*/absl::Seconds(VLOG_IS_ON(2) ? 10 : 60),  //
      /*also_log_timelapses=*/VLOG_IS_ON(10));

  std::vector<Environment> envs_per_thread(env.num_threads, env);
  std::vector<std::vector<size_t>> shard_indices_per_thread(env.num_threads);
  // Prepare per-thread envs and input shard indices.
  for (size_t thread_idx = 0; thread_idx < env.num_threads; ++thread_idx) {
    envs_per_thread[thread_idx].my_shard_index += thread_idx;
    // Shuffle the shards, so that every thread produces different result.
    Rng rng(GetRandomSeed(env.seed + thread_idx));
    auto &shard_indices = shard_indices_per_thread[thread_idx];
    shard_indices.resize(env.total_shards);
    std::iota(shard_indices.begin(), shard_indices.end(), 0);
    std::shuffle(shard_indices.begin(), shard_indices.end(), rng);
  }

  // Run the distillation threads in parallel.
  {
    const size_t num_threads = std::min(env.num_threads, kMaxWritingThreads);
    ThreadPool threads{static_cast<int>(num_threads)};
    for (size_t thread_idx = 0; thread_idx < env.num_threads; ++thread_idx) {
      threads.Schedule(
          [&thread_env = envs_per_thread[thread_idx],
           &thread_shard_indices = shard_indices_per_thread[thread_idx]]() {
            DistillTask(thread_env, thread_shard_indices);
          });
    }
  }  // The writing threads join here.

  return EXIT_SUCCESS;
}

}  // namespace centipede
