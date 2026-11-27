// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ml_drift_delegate/delegate/serialization_program_cache/serialization_program_cache.h"

#include <fcntl.h>  // IWYU pragma: keep b/332641196

#if defined(_WIN32)
#include <io.h>
#include <process.h>
#else
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <sys/clonefile.h>
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "ml_drift_delegate/delegate/serialization_program_cache/serialization_program_cache_schema_generated.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/file_util.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/mmap_handle.h"

namespace ml_drift {

namespace {

// The alignment of each entry in the program cache file.
//
// 64KB was chosen because when the file is mapped on Windows machines, it will
// use MapViewOfFile to map the file. MapViewOfFile uses dwAllocationGranularity
// (default 64KB) as the allocation granularity. By using the same alignment,
// we can avoid allocating extra memory when mapping the file.
// Other operating systems have lower alignment requirements.
constexpr size_t kAlignment = 65536;

bool AlignChecked(size_t offset, size_t alignment, size_t* out) {
  const size_t misalign = offset % alignment;
  const size_t pad = misalign ? alignment - misalign : 0;
  if (offset > std::numeric_limits<size_t>::max() - pad) {
    return false;
  }
  *out = offset + pad;
  return true;
}

std::string JoinPath(absl::string_view path1, absl::string_view path2) {
#if defined(_WIN32)
  char slash = '\\';
#else   // defined(_WIN32)
  char slash = '/';
#endif  // defined(_WIN32)
  if (path1.empty()) return std::string(path2);
  return (path1.back() == slash)
             ? absl::StrCat(path1, path2)
             : absl::StrCat(path1, std::string(1, slash), path2);
}

std::string MakeSiblingTempPath(absl::string_view target_path) {
  static std::atomic<uint64_t> counter{0};
  const uint64_t seq = counter.fetch_add(1, std::memory_order_relaxed);
  const size_t tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
#if defined(_WIN32)
  const unsigned long pid = static_cast<unsigned long>(_getpid());
#else
  const unsigned long pid = static_cast<unsigned long>(getpid());
#endif
  return absl::StrCat(target_path, ".tmp.", pid, ".", tid, ".", seq);
}

bool FsyncFile(int fd) {
#if defined(_WIN32)
  return _commit(fd) == 0;
#else
  if (fsync(fd) == 0) return true;
  // Special file descriptors (e.g. memfd) do not support fsync.
  return errno == EINVAL || errno == ENOTSUP;
#endif
}

void FsyncParentDir(absl::string_view file_path) {
#if !defined(_WIN32)
  std::string dir_path;
  const size_t slash = file_path.find_last_of('/');
  if (slash == absl::string_view::npos) {
    dir_path = ".";
  } else if (slash == 0) {
    dir_path = "/";
  } else {
    dir_path = std::string(file_path.substr(0, slash));
  }
  const int dir_fd = open(dir_path.c_str(), O_RDONLY | O_DIRECTORY);
  if (dir_fd >= 0) {
    (void)fsync(dir_fd);
    close(dir_fd);
  }
#endif
}

absl::Status CopyFileToNewTemp(const std::string& src_path,
                               const std::string& tmp_path,
                               FileDescriptor& out_fd) {
#if defined(__APPLE__)
  if (clonefile(src_path.c_str(), tmp_path.c_str(), 0) == 0) {
    out_fd = FileDescriptor::Open(tmp_path.c_str(), O_RDWR, 0644);
    if (out_fd.IsValid()) {
      return absl::OkStatus();
    }
    const int err = errno;
    std::remove(tmp_path.c_str());
    return absl::InternalError(absl::StrCat(
        "Failed to open cloned program cache temp file: ", strerror(err)));
  }
  if (errno == ENOENT) {
    return absl::NotFoundError("Program cache file does not exist.");
  }
#endif

  FileDescriptor src_fd = FileDescriptor::Open(src_path.c_str(), O_RDONLY);
  if (!src_fd.IsValid()) {
    if (errno == ENOENT) {
      return absl::NotFoundError("Program cache file does not exist.");
    }
    return absl::InternalError(
        absl::StrCat("Failed to open program cache file: ", strerror(errno)));
  }

  FileDescriptor dst_fd = FileDescriptor::Open(
      tmp_path.c_str(), O_CREAT | O_EXCL | O_RDWR,  // NOLINT: b/332641196
      0644);
  if (!dst_fd.IsValid()) {
    return absl::InternalError(absl::StrCat(
        "Failed to create program cache temp file: ", strerror(errno)));
  }

  std::vector<uint8_t> buf(256 * 1024);
  while (true) {
#if defined(_WIN32)
    const int bytes_read =
        _read(src_fd.Value(), buf.data(), static_cast<unsigned int>(buf.size()));
#else
    const ssize_t bytes_read = read(src_fd.Value(), buf.data(), buf.size());
#endif
    if (bytes_read == 0) {
      break;
    }
    if (bytes_read < 0) {
      if (errno == EINTR) continue;
      const int err = errno;
      dst_fd.Close();
      std::remove(tmp_path.c_str());
      return absl::InternalError(
          absl::StrCat("Failed to read program cache file: ", strerror(err)));
    }
    if (!dst_fd.Write(buf.data(), static_cast<size_t>(bytes_read))) {
      const int err = errno;
      dst_fd.Close();
      std::remove(tmp_path.c_str());
      return absl::InternalError(absl::StrCat(
          "Failed to write program cache temp file: ", strerror(err)));
    }
  }
  out_fd = std::move(dst_fd);
  return absl::OkStatus();
}

}  // namespace

SerializationProgramCache::SerializationProgramCache(int fd) : fd_(fd) {}

SerializationProgramCache::SerializationProgramCache(
    absl::string_view file_path)
    : file_path_(file_path) {
  if (!file_path_.empty()) {
    // Open existing cache without creating an empty file on read-only misses;
    // Insert() creates and atomically renames from a sibling temp file.
    fd_ = FileDescriptor::Open(file_path_.c_str(), O_RDWR, 0644);
    if (!fd_.IsValid() && errno != ENOENT) {
      fd_ = FileDescriptor::Open(file_path_.c_str(), O_RDONLY, 0644);
    }
  }
}

SerializationProgramCache::SerializationProgramCache(
    absl::string_view directory_path, absl::string_view model_token)
    : SerializationProgramCache(
          JoinPath(directory_path,
                   absl::StrCat(model_token, "_mldrift_program_cache.bin"))) {}

absl::Status SerializationProgramCache::Insert(uint64_t key,
                                               absl::string_view value) {
  const bool use_atomic = !file_path_.empty();
  if (!use_atomic && !fd_.IsValid()) {
    return absl::InvalidArgumentError("Invalid file descriptor.");
  }

  // For path-backed caches, stage a unique sibling temp file from the latest
  // committed file first so metadata, file size, and data bytes stay consistent
  // and live mappings are never mutated in place.
  std::string tmp_path;
  ScopeGuard cleanup_tmp([&tmp_path] {
    if (!tmp_path.empty()) {
      std::remove(tmp_path.c_str());
    }
  });
  FileDescriptor tmp_fd;
  if (use_atomic) {
    tmp_path = MakeSiblingTempPath(file_path_);
    absl::Status copy_status = CopyFileToNewTemp(file_path_, tmp_path, tmp_fd);
    if (!copy_status.ok() &&
        copy_status.code() != absl::StatusCode::kNotFound) {
      return copy_status;
    }
    if (!tmp_fd.IsValid()) {
      tmp_fd = FileDescriptor::Open(
          tmp_path.c_str(), O_CREAT | O_EXCL | O_RDWR,  // NOLINT: b/332641196
          0644);
      if (!tmp_fd.IsValid()) {
        return absl::InternalError(absl::StrCat(
            "Failed to open program cache temp file: ", strerror(errno)));
      }
    }
  }
  FileDescriptor& target_fd = use_atomic ? tmp_fd : fd_;

  // 1. Read existing metadata if present.
  std::vector<ml_drift::program_cache::schema::CacheEntryT> entries;
  const FileDescriptor::Offset raw_file_size = target_fd.SetPosFromEnd(0);
  if (raw_file_size < 0) {
    return absl::InternalError("Failed to determine program cache file size.");
  }
  const size_t file_size = static_cast<size_t>(raw_file_size);

  size_t header_read_size = std::min(file_size, kAlignment);
  if (header_read_size > 0) {
    if (target_fd.SetPos(0) == -1) {
      return absl::InternalError("Failed to seek to beginning of file.");
    }
    std::vector<uint8_t> header_data(header_read_size);
    if (!target_fd.Read(header_data.data(), header_read_size)) {
      return absl::InternalError("Failed to read header.");
    }

    flatbuffers::Verifier verifier(header_data.data(), header_data.size());
    if (ml_drift::program_cache::schema::VerifyCacheMetadataBuffer(verifier)) {
      auto metadata =
          ml_drift::program_cache::schema::GetCacheMetadata(header_data.data());
      if (metadata->entries()) {
        for (const auto* entry : *metadata->entries()) {
          std::unique_ptr<ml_drift::program_cache::schema::CacheEntryT> entry_t(
              entry->UnPack());
          entries.push_back(*entry_t);
        }
      }
    }
  }

  // 2. Check if key exists.
  bool found = false;
  for (auto& entry : entries) {
    if (entry.key == key) {
      found = true;
      break;
    }
  }

  // 3. Determine write location.
  // Align file size to 64KB for the new buffer.
  size_t write_offset = 0;
  if (!AlignChecked(file_size, kAlignment, &write_offset)) {
    return absl::InternalError("Program cache write offset overflow.");
  }
  // Ensure we start at least at 64KB.
  if (write_offset < kAlignment) write_offset = kAlignment;

  // 4. Update metadata structure and serialize before mutating disk state.
  if (!found) {
    ml_drift::program_cache::schema::CacheEntryT new_entry;
    new_entry.key = key;
    entries.push_back(new_entry);
  }

  for (auto& entry : entries) {
    if (entry.key == key) {
      entry.offset = write_offset;
      entry.size = value.size();
      break;
    }
  }

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<ml_drift::program_cache::schema::CacheEntry>>
      entry_offsets;
  entry_offsets.reserve(entries.size());
  for (const auto& entry : entries) {
    entry_offsets.push_back(ml_drift::program_cache::schema::CreateCacheEntry(
        builder, entry.key, entry.offset, entry.size));
  }
  auto entries_vec = builder.CreateVector(entry_offsets);
  auto metadata_offset = ml_drift::program_cache::schema::CreateCacheMetadata(
      builder, entries_vec);
  builder.Finish(metadata_offset);

  if (builder.GetSize() > kAlignment) {
    return absl::ResourceExhaustedError("Metadata size exceeds 64KB limit.");
  }

  // 5. Write value, metadata, fsync, and atomically rename when path-backed.
  if (target_fd.SetPos(write_offset) == -1) {
    return absl::InternalError("Failed to seek to write offset.");
  }
  if (!target_fd.Write(value.data(), value.size())) {
    return absl::InternalError("Failed to write value.");
  }
  if (!use_atomic) {
    // Raw file descriptor path: flush payload before updating offset-0 header.
    (void)FsyncFile(target_fd.Value());
  }
  if (target_fd.SetPos(0) == -1) {
    return absl::InternalError("Failed to seek to beginning.");
  }
  if (!target_fd.Write(builder.GetBufferPointer(), builder.GetSize())) {
    return absl::InternalError("Failed to write metadata.");
  }
  if (!FsyncFile(target_fd.Value())) {
    return absl::InternalError(
        absl::StrCat("Failed to fsync program cache file: ", strerror(errno)));
  }

  if (use_atomic) {
    if (std::rename(tmp_path.c_str(), file_path_.c_str()) != 0) {
      return absl::InternalError(absl::StrCat(
          "Failed to rename program cache temp file: ", strerror(errno)));
    }
    tmp_path.clear();
    FsyncParentDir(file_path_);
    fd_ = std::move(tmp_fd);
  }

  return absl::OkStatus();
}

absl::StatusOr<MMapHandle> SerializationProgramCache::LookUpHandle(
    uint64_t key) {
  if (!fd_.IsValid()) {
    if (!file_path_.empty()) {
      fd_ = FileDescriptor::Open(file_path_.c_str(), O_RDONLY, 0644);
      if (!fd_.IsValid()) {
        if (errno == ENOENT) {
          return absl::NotFoundError("Program cache file does not exist.");
        }
        return absl::InvalidArgumentError("Invalid file descriptor.");
      }
    } else {
      return absl::InvalidArgumentError("Invalid file descriptor.");
    }
  }

  const FileDescriptor::Offset raw_file_size = fd_.SetPosFromEnd(0);
  if (raw_file_size < 0) {
    return absl::InternalError("Failed to determine program cache file size.");
  }
  const size_t file_size = static_cast<size_t>(raw_file_size);

  MMapHandle mmap_handle;
  // Map up to kAlignment, or file_size if smaller.
  size_t map_size = std::min(file_size, kAlignment);
  if (map_size == 0) {
    return absl::NotFoundError("File is empty.");
  }

  ABSL_RETURN_IF_ERROR(mmap_handle.Map(fd_, 0, map_size));

  flatbuffers::Verifier verifier(mmap_handle.data(), mmap_handle.size());
  if (!ml_drift::program_cache::schema::VerifyCacheMetadataBuffer(verifier)) {
    return absl::InternalError("Corrupt metadata.");
  }

  auto metadata =
      ml_drift::program_cache::schema::GetCacheMetadata(mmap_handle.data());
  uint64_t found_offset = 0;
  uint64_t found_size = 0;
  bool found = false;

  if (metadata->entries()) {
    for (const auto* entry : *metadata->entries()) {
      if (entry->key() == key) {
        found_offset = entry->offset();
        found_size = entry->size();
        found = true;
        break;
      }
    }
  }

  if (!found) {
    return absl::NotFoundError(absl::StrCat("Key ", key, " not found."));
  }

  if (found_offset > file_size || found_size > file_size - found_offset) {
    return absl::InternalError("Data offset/size out of bounds.");
  }

  MMapHandle data_handle;
  ABSL_RETURN_IF_ERROR(data_handle.Map(fd_, found_offset, found_size));

  return data_handle;
}

absl::StatusOr<std::string> SerializationProgramCache::LookUp(uint64_t key) {
  ABSL_ASSIGN_OR_RETURN(auto handle, LookUpHandle(key));
  return std::string(reinterpret_cast<const char*>(handle.data()),
                     handle.size());
}

}  // namespace ml_drift
