// C_package/fast_subgraphs.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <stdexcept>
#include <cstdint>

namespace py = pybind11;

class DSU {
public:
    std::vector<int> parent;
    std::vector<unsigned char> rank;

    explicit DSU(int n) {
        parent.resize(n);
        rank.assign(n, 0);
        for (int i = 0; i < n; ++i) parent[i] = i;
    }

    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    void unite(int a, int b) {
        int ra = find(a);
        int rb = find(b);
        if (ra == rb) return;

        if (rank[ra] < rank[rb]) {
            parent[ra] = rb;
        } else if (rank[ra] > rank[rb]) {
            parent[rb] = ra;
        } else {
            parent[rb] = ra;
            rank[ra]++;
        }
    }
};


py::tuple build_subgraphs_cpp(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> frames1_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> parents1_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> frames2_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> parents2_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> matches_arr
) {
    auto frames1 = frames1_arr.unchecked<1>();
    auto parents1 = parents1_arr.unchecked<1>();
    auto frames2 = frames2_arr.unchecked<1>();
    auto parents2 = parents2_arr.unchecked<1>();

    const int n1 = static_cast<int>(frames1.shape(0));
    const int n2 = static_cast<int>(frames2.shape(0));
    const int N = n1 + n2;

    if (parents1.shape(0) != n1) {
        throw std::runtime_error("parents1 and frames1 must have the same length.");
    }
    if (parents2.shape(0) != n2) {
        throw std::runtime_error("parents2 and frames2 must have the same length.");
    }

    auto m2 = [n1](int j) {
        return n1 + j;
    };

    DSU dsu(N);

    std::vector<std::vector<int>> dir_edges(N);
    std::vector<std::vector<int>> children1(n1);
    std::vector<std::vector<int>> children2(n2);

    // movie1 parent-child edges
    for (int i = 0; i < n1; ++i) {
        int p = static_cast<int>(parents1(i));
        if (p != -1) {
            if (p < 0 || p >= n1) {
                throw std::runtime_error("Invalid parent index in parents1.");
            }
            children1[p].push_back(i);
            dir_edges[p].push_back(i);
            dsu.unite(p, i);
        }
    }

    // movie2 parent-child edges
    for (int j = 0; j < n2; ++j) {
        int p = static_cast<int>(parents2(j));
        if (p != -1) {
            if (p < 0 || p >= n2) {
                throw std::runtime_error("Invalid parent index in parents2.");
            }
            children2[p].push_back(j);

            int src = m2(p);
            int dst = m2(j);
            dir_edges[src].push_back(dst);
            dsu.unite(src, dst);
        }
    }

    // matches: shape should be (M, >=2)
    if (matches_arr.ndim() != 2 || matches_arr.shape(1) < 2) {
        throw std::runtime_error("matches must be a 2D array with at least 2 columns.");
    }

    auto matches = matches_arr.unchecked<2>();
    const int M = static_cast<int>(matches.shape(0));

    for (int k = 0; k < M; ++k) {
        int i1 = static_cast<int>(matches(k, 0));
        int i2 = static_cast<int>(matches(k, 1));

        if (i1 < 0 || i1 >= n1 || i2 < 0 || i2 >= n2) {
            throw std::runtime_error("Invalid match index.");
        }

        int u = i1;
        int v = m2(i2);

        // direct match edge
        dsu.unite(u, v);

        int p1 = static_cast<int>(parents1(i1));
        int p2 = static_cast<int>(parents2(i2));

        // v connects to parent of i1
        if (p1 != -1) {
            dsu.unite(v, p1);
            dir_edges[p1].push_back(v);
        }

        // u connects to parent of i2
        if (p2 != -1) {
            int p2_global = m2(p2);
            dsu.unite(u, p2_global);
            dir_edges[p2_global].push_back(u);
        }

        // v connects to children of i1
        for (int c1 : children1[i1]) {
            dsu.unite(v, c1);
            dir_edges[v].push_back(c1);
        }

        // u connects to children of i2
        for (int c2 : children2[i2]) {
            int c2_global = m2(c2);
            dsu.unite(u, c2_global);
            dir_edges[u].push_back(c2_global);
        }
    }

    // Deduplicate directed edges
    for (auto &nbrs : dir_edges) {
        if (nbrs.size() > 1) {
            std::sort(nbrs.begin(), nbrs.end());
            nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
        }
    }

    // Collect connected components
    std::unordered_map<int, std::vector<int>> comp_map;
    comp_map.reserve(N);

    for (int u = 0; u < N; ++u) {
        int r = dsu.find(u);
        comp_map[r].push_back(u);
    }

    std::vector<std::vector<int>> easy_subgraphs;
    std::vector<std::vector<int>> hard_subgraphs;
    easy_subgraphs.reserve(comp_map.size());
    hard_subgraphs.reserve(comp_map.size());

    // Classify easy / hard
    for (auto &kv : comp_map) {
        const std::vector<int> &nodes = kv.second;

        std::unordered_set<int64_t> seen_frame1;
        std::unordered_set<int64_t> seen_frame2;
        seen_frame1.reserve(nodes.size());
        seen_frame2.reserve(nodes.size());

        bool easy = true;

        for (int u : nodes) {
            if (u < n1) {
                int64_t f = frames1(u);
                if (seen_frame1.find(f) != seen_frame1.end()) {
                    easy = false;
                    break;
                }
                seen_frame1.insert(f);
            } else {
                int j = u - n1;
                int64_t f = frames2(j);
                if (seen_frame2.find(f) != seen_frame2.end()) {
                    easy = false;
                    break;
                }
                seen_frame2.insert(f);
            }
        }

        if (easy) {
            easy_subgraphs.push_back(nodes);
        } else {
            hard_subgraphs.push_back(nodes);
        }
    }

    return py::make_tuple(easy_subgraphs, hard_subgraphs, dir_edges);
}


PYBIND11_MODULE(fast_subgraphs, m) {
    m.doc() = "Fast C++ implementation of build_subgraphs for fusion.";
    m.def(
        "build_subgraphs_cpp",
        &build_subgraphs_cpp,
        py::arg("frames1"),
        py::arg("parents1"),
        py::arg("frames2"),
        py::arg("parents2"),
        py::arg("matches")
    );
}