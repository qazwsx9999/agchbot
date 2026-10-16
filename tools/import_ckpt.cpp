// Import a converted Python checkpoint (weights.bin + meta.txt) into a native
// C++ engine checkpoint (checkpoint.pt) that `alphazero_chess --resume` reads.
//
//   import_ckpt <weights.bin> <meta.txt> <out_checkpoint.pt>
//
// weights.bin layout (little-endian, produced by tools/convert_checkpoint.py):
//   int32  num_tensors
//   repeat:
//     int32 name_len; char name[name_len]
//     int32 ndim; int32 dims[ndim]
//     float32 data[prod(dims)]
//
// The parameter/buffer names match the module hierarchy exactly
// (conv_in.weight, res_tower.0.conv1.weight, value_fc2.bias, ...), so each
// tensor is copied into the model by name. SGD momentum is NOT imported; the
// engine simply starts training again with fresh momentum.

#include "model.hpp"
#include "config.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::string read_string(std::ifstream& f) {
    int32_t len = 0;
    f.read(reinterpret_cast<char*>(&len), sizeof(len));
    std::string s(static_cast<size_t>(len), '\0');
    f.read(&s[0], len);
    return s;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: import_ckpt <weights.bin> <meta.txt> <out_checkpoint.pt>\n";
        return 1;
    }

    az::Config cfg; // default architecture (10 res blocks, 128 filters) matches the checkpoint
    az::ChessNet model(cfg);
    model->eval();

    auto params = model->named_parameters();
    auto buffers = model->named_buffers();

    std::ifstream f(argv[1], std::ios::binary);
    if (!f) { std::cerr << "cannot open " << argv[1] << "\n"; return 1; }

    int32_t num = 0;
    f.read(reinterpret_cast<char*>(&num), sizeof(num));

    torch::NoGradGuard no_grad;
    int copied = 0, skipped = 0;

    for (int i = 0; i < num; ++i) {
        std::string name = read_string(f);
        int32_t ndim = 0;
        f.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        std::vector<int64_t> shape(static_cast<size_t>(ndim));
        int64_t numel = 1;
        for (int d = 0; d < ndim; ++d) {
            int32_t s = 0;
            f.read(reinterpret_cast<char*>(&s), sizeof(s));
            shape[static_cast<size_t>(d)] = s;
            numel *= s;
        }
        std::vector<float> data(static_cast<size_t>(numel));
        f.read(reinterpret_cast<char*>(data.data()), numel * static_cast<std::streamsize>(sizeof(float)));

        torch::Tensor* dst = params.find(name);
        if (dst == nullptr) dst = buffers.find(name);
        if (dst == nullptr) {
            std::cerr << "  [skip] no match for '" << name << "'\n";
            ++skipped;
            continue;
        }
        if (dst->numel() != numel) {
            std::cerr << "  [skip] numel mismatch for '" << name << "' ("
                      << dst->numel() << " vs " << numel << ")\n";
            ++skipped;
            continue;
        }
        torch::Tensor src = torch::from_blob(
            data.data(), shape, torch::TensorOptions().dtype(torch::kFloat32)).clone();
        dst->copy_(src.reshape(dst->sizes()).to(dst->dtype()));
        ++copied;
    }

    int iteration = 0, total_games = 0;
    {
        std::ifstream m(argv[2]);
        std::string key;
        while (m >> key) {
            if (key == "iteration") m >> iteration;
            else if (key == "total_games") m >> total_games;
        }
    }

    torch::serialize::OutputArchive archive;
    model->save(archive);
    archive.write("iteration", torch::tensor(iteration));
    archive.write("total_games", torch::tensor(total_games));
    archive.save_to(argv[3]);

    std::cout << "Imported " << copied << " tensors (" << skipped << " skipped).\n";
    std::cout << "Wrote " << argv[3] << " (iteration=" << iteration
              << ", total_games=" << total_games << ").\n";
    std::cout << "Place this next to replay_buffer.bin in your --checkpoint-dir and run with --resume.\n";
    return 0;
}
