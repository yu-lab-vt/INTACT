#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;


// ============================================================
// 1. Basic geometry utilities
// ============================================================

struct Point3i {
    int z;
    int y;
    int x;
};

static inline int64_t idx3d(int z, int y, int x, int Y, int X) {
    return (static_cast<int64_t>(z) * Y + y) * X + x;
}

static inline int64_t c_index_3d(
    int64_t z,
    int64_t y,
    int64_t x,
    int64_t Y,
    int64_t X
) {
    return (z * Y + y) * X + x;
}

/*
 * New unified voxel reader.
 *
 * Python new behavior:
 *   2D vox: [y, x] -> pseudo-3D [z=0, y, x]
 *   3D vox: [z, y, x]
 *
 * Important:
 *   vox_flat is packed as movie1 vox + movie2 vox.
 *   Therefore merged global node id can directly index offsets:
 *       movie1 node u      -> cell_id = u
 *       movie2 node n1 + j -> cell_id = n1 + j
 */
static std::vector<Point3i> get_cell_voxels_as_zyx_ptr(
    const int32_t* vox,
    const int64_t* offsets,
    int64_t cell_id,
    int dim
) {
    if (cell_id < 0) {
        throw std::runtime_error("Negative cell_id in get_cell_voxels_as_zyx_ptr.");
    }

    const int64_t start = offsets[cell_id];
    const int64_t end = offsets[cell_id + 1];

    if (end < start) {
        throw std::runtime_error("Invalid voxel offsets: end < start.");
    }

    std::vector<Point3i> pts;
    pts.reserve(static_cast<size_t>(end - start));

    for (int64_t i = start; i < end; ++i) {
        if (dim == 2) {
            /*
             * Python:
             *   original 2D vox is [y, x]
             *   lifted to [z=0, y, x]
             */
            int y = static_cast<int>(vox[2 * i]);
            int x = static_cast<int>(vox[2 * i + 1]);
            pts.push_back(Point3i{0, y, x});
        } else if (dim == 3) {
            /*
             * Python convention:
             *   3D vox is already [z, y, x]
             */
            int z = static_cast<int>(vox[3 * i]);
            int y = static_cast<int>(vox[3 * i + 1]);
            int x = static_cast<int>(vox[3 * i + 2]);
            pts.push_back(Point3i{z, y, x});
        } else {
            throw std::runtime_error("Unsupported voxel dimension. Expected dim=2 or dim=3.");
        }
    }

    return pts;
}


// ============================================================
// 2. External C functions
// ============================================================

extern "C" {
    void edt_3d(const unsigned char* ref_cell,
                const int* s1_dims,
                int s1_ndims,
                const unsigned char* mov_cell,
                const int* s2_dims,
                int s2_ndims,
                const float* shift,
                float* output);

    typedef long long price_t;

    price_t* pyCS2(long* msz,
                   double* mtail,
                   double* mhead,
                   double* mlow,
                   double* macap,
                   double* mcost);

    void pyFreeTrackVec(price_t* track_vec);
}


// ============================================================
// 3. New EDT-based region distance
// ============================================================

/*
 * This implements Python ovDistanceRegion(...)[0] for the default branch:
 *
 *   - 2D has already been lifted to pseudo-3D before entering this function.
 *   - 3D is directly used.
 *   - If either region has fewer than 2 voxels, return 100.0.
 *   - Build local bbox masks.
 *   - Call edt_3d twice:
 *       dist2cell1 = edt_3d(mask1, mask2, mov_shift)
 *       dist2cell2 = edt_3d(mask2, mask1, -mov_shift)
 *   - Take sqrt of squared distances.
 *   - Return max(mean(cur->next), mean(next->cur)).
 */
static double region_distance_edt_max_zyx(
    const std::vector<Point3i>& cur_pts,
    const std::vector<Point3i>& next_pts
) {
    if (cur_pts.size() < 2 || next_pts.size() < 2) {
        return 100.0;
    }

    auto get_bbox = [](const std::vector<Point3i>& pts) {
        std::array<int, 3> mn = {pts[0].z, pts[0].y, pts[0].x};
        std::array<int, 3> mx = {pts[0].z, pts[0].y, pts[0].x};

        for (const auto& p : pts) {
            mn[0] = std::min(mn[0], p.z);
            mn[1] = std::min(mn[1], p.y);
            mn[2] = std::min(mn[2], p.x);

            mx[0] = std::max(mx[0], p.z);
            mx[1] = std::max(mx[1], p.y);
            mx[2] = std::max(mx[2], p.x);
        }

        return std::make_pair(mn, mx);
    };

    auto bbox1 = get_bbox(cur_pts);
    auto bbox2 = get_bbox(next_pts);

    auto st1 = bbox1.first;
    auto ed1 = bbox1.second;
    auto st2 = bbox2.first;
    auto ed2 = bbox2.second;

    int Z1 = ed1[0] - st1[0] + 1;
    int Y1 = ed1[1] - st1[1] + 1;
    int X1 = ed1[2] - st1[2] + 1;

    int Z2 = ed2[0] - st2[0] + 1;
    int Y2 = ed2[1] - st2[1] + 1;
    int X2 = ed2[2] - st2[2] + 1;

    if (Z1 <= 0 || Y1 <= 0 || X1 <= 0 ||
        Z2 <= 0 || Y2 <= 0 || X2 <= 0) {
        throw std::runtime_error("Invalid bounding box size in region_distance_edt_max_zyx.");
    }

    const int64_t sz1 = static_cast<int64_t>(Z1) * Y1 * X1;
    const int64_t sz2 = static_cast<int64_t>(Z2) * Y2 * X2;

    if (sz1 <= 0 || sz2 <= 0) {
        return 100.0;
    }

    std::vector<unsigned char> mask1(static_cast<size_t>(sz1), 0);
    std::vector<unsigned char> mask2(static_cast<size_t>(sz2), 0);

    std::vector<Point3i> cell1_sub;
    std::vector<Point3i> cell2_sub;

    cell1_sub.reserve(cur_pts.size());
    cell2_sub.reserve(next_pts.size());

    for (const auto& p : cur_pts) {
        int z = p.z - st1[0];
        int y = p.y - st1[1];
        int x = p.x - st1[2];

        mask1[static_cast<size_t>(idx3d(z, y, x, Y1, X1))] = 1;
        cell1_sub.push_back(Point3i{z, y, x});
    }

    for (const auto& p : next_pts) {
        int z = p.z - st2[0];
        int y = p.y - st2[1];
        int x = p.x - st2[2];

        mask2[static_cast<size_t>(idx3d(z, y, x, Y2, X2))] = 1;
        cell2_sub.push_back(Point3i{z, y, x});
    }

    /*
     * Python:
     *   mov_shift = st_pt2 - st_pt1 - frame_shift
     *
     * In easy fusion, frame_shift is not provided, so it is [0,0,0].
     */
    float shift12[3] = {
        static_cast<float>(st2[0] - st1[0]),
        static_cast<float>(st2[1] - st1[1]),
        static_cast<float>(st2[2] - st1[2])
    };

    float shift21[3] = {
        -shift12[0],
        -shift12[1],
        -shift12[2]
    };

    int dims1[3] = {Z1, Y1, X1};
    int dims2[3] = {Z2, Y2, X2};

    std::vector<float> dist2cell1(static_cast<size_t>(sz2), 0.0f);
    std::vector<float> dist2cell2(static_cast<size_t>(sz1), 0.0f);

    edt_3d(
        mask1.data(),
        dims1,
        3,
        mask2.data(),
        dims2,
        3,
        shift12,
        dist2cell1.data()
    );

    edt_3d(
        mask2.data(),
        dims2,
        3,
        mask1.data(),
        dims1,
        3,
        shift21,
        dist2cell2.data()
    );

    double sum_next_to_cur = 0.0;
    for (const auto& p : cell2_sub) {
        float d2 = dist2cell1[static_cast<size_t>(idx3d(p.z, p.y, p.x, Y2, X2))];
        sum_next_to_cur += std::sqrt(static_cast<double>(std::max(d2, 0.0f)));
    }

    double sum_cur_to_next = 0.0;
    for (const auto& p : cell1_sub) {
        float d2 = dist2cell2[static_cast<size_t>(idx3d(p.z, p.y, p.x, Y1, X1))];
        sum_cur_to_next += std::sqrt(static_cast<double>(std::max(d2, 0.0f)));
    }

    const double mean_cur_to_next =
        sum_cur_to_next / static_cast<double>(cell1_sub.size());

    const double mean_next_to_cur =
        sum_next_to_cur / static_cast<double>(cell2_sub.size());

    return std::max(mean_cur_to_next, mean_next_to_cur);
}

/*
 * New edge cost.
 *
 * Old behavior:
 *   dim == 2 -> overlap/Jaccard cost
 *   dim == 3 -> EDT cost
 *
 * New Python behavior:
 *   dim == 2 -> lift [y,x] to [z=0,y,x], then EDT
 *   dim == 3 -> EDT
 */
static double edge_cost_cpp(
    int u,
    int v,
    const int32_t* vox,
    const int64_t* offsets,
    int dim
) {
    if (dim != 2 && dim != 3) {
        throw std::runtime_error("Only 2D or 3D voxels are supported.");
    }

    std::vector<Point3i> pts_u = get_cell_voxels_as_zyx_ptr(
        vox,
        offsets,
        static_cast<int64_t>(u),
        dim
    );

    std::vector<Point3i> pts_v = get_cell_voxels_as_zyx_ptr(
        vox,
        offsets,
        static_cast<int64_t>(v),
        dim
    );

    return region_distance_edt_max_zyx(pts_u, pts_v);
}


// ============================================================
// 4. Subgraph construction
// ============================================================

struct DSU {
    std::vector<int> parent;
    std::vector<unsigned char> rank;

    explicit DSU(int n) : parent(n), rank(n, 0) {
        std::iota(parent.begin(), parent.end(), 0);
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

struct DetArc {
    double id;
    double c_en;
    double c_ex;
    double c_det;
};

struct TransArc {
    double src;
    double dst;
    double cost;
};

struct CostJob {
    size_t arc_pos;
    int u;
    int v;
};

static void build_subgraphs_internal(
    const py::detail::unchecked_reference<int64_t, 1>& frames1,
    const py::detail::unchecked_reference<int64_t, 1>& parents1,
    const py::detail::unchecked_reference<int64_t, 1>& frames2,
    const py::detail::unchecked_reference<int64_t, 1>& parents2,
    const py::detail::unchecked_reference<int64_t, 2>& matches,
    std::vector<std::vector<int>>& easy_subgraphs,
    std::vector<std::vector<int>>& hard_subgraphs,
    std::vector<std::vector<int>>& dir_edges
) {
    const int n1 = static_cast<int>(frames1.shape(0));
    const int n2 = static_cast<int>(frames2.shape(0));
    const int N = n1 + n2;

    auto m2 = [n1](int j) {
        return n1 + j;
    };

    DSU dsu(N);
    dir_edges.assign(N, {});

    std::vector<std::vector<int>> children1(n1);
    std::vector<std::vector<int>> children2(n2);

    /*
     * Intra-movie edges.
     */
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

    /*
     * Cross-movie edges induced by matches.
     *
     * This matches Python build_subgraphs:
     *   - u and v are connected for undirected connectivity.
     *   - parent/child cross edges are also added.
     *   - direct match edge itself is not added into dir_edges.
     */
    const int M = static_cast<int>(matches.shape(0));

    for (int k = 0; k < M; ++k) {
        int i1 = static_cast<int>(matches(k, 0));
        int i2 = static_cast<int>(matches(k, 1));

        if (i1 < 0 || i1 >= n1 || i2 < 0 || i2 >= n2) {
            throw std::runtime_error("Invalid match index.");
        }

        int u = i1;
        int v = m2(i2);

        dsu.unite(u, v);

        int p1 = static_cast<int>(parents1(i1));
        int p2 = static_cast<int>(parents2(i2));

        if (p1 != -1) {
            dsu.unite(v, p1);
            dir_edges[p1].push_back(v);
        }

        if (p2 != -1) {
            int p2g = m2(p2);
            dsu.unite(u, p2g);
            dir_edges[p2g].push_back(u);
        }

        for (int c1 : children1[i1]) {
            dsu.unite(v, c1);
            dir_edges[v].push_back(c1);
        }

        for (int c2 : children2[i2]) {
            int c2g = m2(c2);
            dsu.unite(u, c2g);
            dir_edges[u].push_back(c2g);
        }
    }

    /*
     * Remove duplicate directed edges.
     */
    for (auto& nbrs : dir_edges) {
        if (nbrs.size() > 1) {
            std::sort(nbrs.begin(), nbrs.end());
            nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
        }
    }

    /*
     * Extract connected components from DSU.
     */
    std::vector<int> root_to_comp(N, -1);
    std::vector<std::vector<int>> comps;
    comps.reserve(N);

    for (int u = 0; u < N; ++u) {
        int r = dsu.find(u);

        if (root_to_comp[r] < 0) {
            root_to_comp[r] = static_cast<int>(comps.size());
            comps.emplace_back();
        }

        comps[root_to_comp[r]].push_back(u);
    }

    /*
     * Easy / hard classification:
     * A component is hard if one movie has more than one node in the same frame.
     * movie1 and movie2 are counted separately.
     */
    int64_t min_f1 = std::numeric_limits<int64_t>::max();
    int64_t max_f1 = std::numeric_limits<int64_t>::min();
    int64_t min_f2 = std::numeric_limits<int64_t>::max();
    int64_t max_f2 = std::numeric_limits<int64_t>::min();

    for (int i = 0; i < n1; ++i) {
        min_f1 = std::min(min_f1, frames1(i));
        max_f1 = std::max(max_f1, frames1(i));
    }

    for (int j = 0; j < n2; ++j) {
        min_f2 = std::min(min_f2, frames2(j));
        max_f2 = std::max(max_f2, frames2(j));
    }

    int64_t off1 = min_f1;
    int64_t off2 = min_f2;

    size_t len1 = static_cast<size_t>(max_f1 - min_f1 + 1);
    size_t len2 = static_cast<size_t>(max_f2 - min_f2 + 1);

    std::vector<int> seen1(len1, -1);
    std::vector<int> seen2(len2, -1);

    int stamp = 0;

    for (const auto& nodes : comps) {
        ++stamp;
        bool easy = true;

        for (int u : nodes) {
            if (u < n1) {
                size_t idx = static_cast<size_t>(frames1(u) - off1);

                if (seen1[idx] == stamp) {
                    easy = false;
                    break;
                }

                seen1[idx] = stamp;
            } else {
                int j = u - n1;
                size_t idx = static_cast<size_t>(frames2(j) - off2);

                if (seen2[idx] == stamp) {
                    easy = false;
                    break;
                }

                seen2[idx] = stamp;
            }
        }

        if (easy) {
            easy_subgraphs.push_back(nodes);
        } else {
            hard_subgraphs.push_back(nodes);
        }
    }
}


// ============================================================
// 5. Easy graph arcs
// ============================================================

static void build_easy_arcs(
    const std::vector<std::vector<int>>& easy_subgraphs,
    const std::vector<std::vector<int>>& edges,
    const py::detail::unchecked_reference<int64_t, 1>& frames1,
    const py::detail::unchecked_reference<int64_t, 1>& frames2,
    const int32_t* vox,
    const int64_t* offsets,
    int dim,
    double threshold,
    std::vector<DetArc>& detection_arcs,
    std::vector<TransArc>& transition_arcs
) {
    const int n1 = static_cast<int>(frames1.shape(0));
    const int n2 = static_cast<int>(frames2.shape(0));
    const int N = n1 + n2;

    if (N <= 0) {
        detection_arcs.clear();
        transition_arcs.clear();
        return;
    }

    int64_t min_frame1 = frames1(0);
    int64_t max_frame2 = frames2(0);

    for (int i = 0; i < n1; ++i) {
        min_frame1 = std::min(min_frame1, frames1(i));
    }

    for (int j = 0; j < n2; ++j) {
        max_frame2 = std::max(max_frame2, frames2(j));
    }

    const double threshold2 = -2.0 * threshold - 0.00001;

    const double big =
        (-(static_cast<double>(max_frame2 - min_frame1 + 1)) * threshold2) + 1.0;

    detection_arcs.clear();
    detection_arcs.resize(static_cast<size_t>(N + easy_subgraphs.size()));

    /*
     * Original nodes.
     */
    for (int u = 0; u < N; ++u) {
        detection_arcs[static_cast<size_t>(u)] =
            DetArc{
                static_cast<double>(u),
                big,
                2.0 * threshold,
                threshold2
            };
    }

    /*
     * One control node per easy subgraph.
     */
    for (size_t gi = 0; gi < easy_subgraphs.size(); ++gi) {
        int ctrl = N + static_cast<int>(gi);

        detection_arcs[static_cast<size_t>(ctrl)] =
            DetArc{
                static_cast<double>(ctrl),
                threshold,
                threshold,
                threshold2
            };
    }

    /*
     * Reserve transition arcs.
     */
    transition_arcs.clear();

    size_t approx_edges = 0;

    for (const auto& sg : easy_subgraphs) {
        for (int u : sg) {
            if (u >= 0 && u < static_cast<int>(edges.size())) {
                approx_edges += edges[static_cast<size_t>(u)].size();
            }
        }
    }

    transition_arcs.reserve(approx_edges + easy_subgraphs.size() * 2);

    std::vector<CostJob> cost_jobs;
    cost_jobs.reserve(approx_edges);

    /*
     * Build arcs in the same order as Python pruning.
     */
    for (size_t gi = 0; gi < easy_subgraphs.size(); ++gi) {
        const auto& sg = easy_subgraphs[gi];

        if (sg.empty()) {
            continue;
        }

        int64_t minf = std::numeric_limits<int64_t>::max();
        int64_t maxf = std::numeric_limits<int64_t>::min();

        for (int u : sg) {
            int64_t f = (u < n1) ? frames1(u) : frames2(u - n1);
            minf = std::min(minf, f);
            maxf = std::max(maxf, f);
        }

        const int ctrl = N + static_cast<int>(gi);

        /*
         * Real transition arcs.
         */
        for (int u : sg) {
            int64_t f = (u < n1) ? frames1(u) : frames2(u - n1);

            if (f == maxf) {
                detection_arcs[static_cast<size_t>(u)].c_ex = threshold;
            }

            if (u < 0 || u >= static_cast<int>(edges.size())) {
                continue;
            }

            const auto& nbrs = edges[static_cast<size_t>(u)];

            for (int v : nbrs) {
                const size_t pos = transition_arcs.size();

                transition_arcs.push_back(
                    TransArc{
                        static_cast<double>(u),
                        static_cast<double>(v),
                        0.0
                    }
                );

                cost_jobs.push_back(
                    CostJob{
                        pos,
                        u,
                        v
                    }
                );
            }
        }

        /*
         * Control arcs.
         *
         * Python order:
         *   first_frame_nodes = concat([min_idx_in_part1, min_idx_in_part2])
         * Therefore movie1 first-frame nodes first, then movie2 first-frame nodes.
         */
        for (int u : sg) {
            if (u < n1) {
                int64_t f = frames1(u);

                if (f == minf) {
                    transition_arcs.push_back(
                        TransArc{
                            static_cast<double>(ctrl),
                            static_cast<double>(u),
                            0.0
                        }
                    );
                }
            }
        }

        for (int u : sg) {
            if (u >= n1) {
                int j = u - n1;
                int64_t f = frames2(j);

                if (f == minf) {
                    transition_arcs.push_back(
                        TransArc{
                            static_cast<double>(ctrl),
                            static_cast<double>(u),
                            0.0
                        }
                    );
                }
            }
        }
    }

    /*
     * Compute transition costs.
     *
     * The order of transition_arcs is preserved because each job writes back
     * to its original arc_pos.
     */
    #pragma omp parallel for schedule(dynamic)
    for (long long k = 0; k < static_cast<long long>(cost_jobs.size()); ++k) {
        const CostJob& job = cost_jobs[static_cast<size_t>(k)];

        const double c = edge_cost_cpp(
            job.u,
            job.v,
            vox,
            offsets,
            dim
        );

        transition_arcs[job.arc_pos].cost = c;
    }
}


// ============================================================
// 6. CINDA / CS2 solver wrapper
// ============================================================

static std::vector<std::vector<int>> run_cinda_cpp(
    const std::vector<DetArc>& detection_arcs,
    const std::vector<TransArc>& transition_arcs,
    int N_original
) {
    const long n_detection = static_cast<long>(detection_arcs.size());
    const long n_transition = static_cast<long>(transition_arcs.size());
    const long n_arcs = n_detection * 3 + n_transition;

    std::vector<double> mtail(static_cast<size_t>(n_arcs));
    std::vector<double> mhead(static_cast<size_t>(n_arcs));
    std::vector<double> mlow(static_cast<size_t>(n_arcs), 0.0);
    std::vector<double> macap(static_cast<size_t>(n_arcs), 1.0);
    std::vector<double> mcost(static_cast<size_t>(n_arcs));

    long pos = 0;

    /*
     * Entry arcs.
     */
    for (long i = 0; i < n_detection; ++i, ++pos) {
        double did = detection_arcs[static_cast<size_t>(i)].id + 1.0;

        mtail[static_cast<size_t>(pos)] = 1.0;
        mhead[static_cast<size_t>(pos)] = did * 2.0;
        mcost[static_cast<size_t>(pos)] =
            detection_arcs[static_cast<size_t>(i)].c_en;
    }

    /*
     * Exit arcs.
     */
    for (long i = 0; i < n_detection; ++i, ++pos) {
        double did = detection_arcs[static_cast<size_t>(i)].id + 1.0;

        mtail[static_cast<size_t>(pos)] = did * 2.0 + 1.0;
        mhead[static_cast<size_t>(pos)] = 1.0;
        mcost[static_cast<size_t>(pos)] =
            detection_arcs[static_cast<size_t>(i)].c_ex;
    }

    /*
     * Detection arcs.
     */
    for (long i = 0; i < n_detection; ++i, ++pos) {
        double did = detection_arcs[static_cast<size_t>(i)].id + 1.0;

        mtail[static_cast<size_t>(pos)] = did * 2.0;
        mhead[static_cast<size_t>(pos)] = did * 2.0 + 1.0;
        mcost[static_cast<size_t>(pos)] =
            detection_arcs[static_cast<size_t>(i)].c_det;
    }

    /*
     * Transition arcs.
     */
    for (long i = 0; i < n_transition; ++i, ++pos) {
        double s = transition_arcs[static_cast<size_t>(i)].src + 1.0;
        double t = transition_arcs[static_cast<size_t>(i)].dst + 1.0;

        mtail[static_cast<size_t>(pos)] = s * 2.0 + 1.0;
        mhead[static_cast<size_t>(pos)] = t * 2.0;
        mcost[static_cast<size_t>(pos)] =
            transition_arcs[static_cast<size_t>(i)].cost;
    }

    /*
     * Match Python mcc4mot behavior:
     *
     * Python:
     *   if isinstance(mcost[0], float):
     *       mcost = [int(n * 10**7) for n in mcost]
     *
     * C++ static_cast<long long> truncates toward zero, same as Python int().
     */
    constexpr double SCALE = 1e7;

    for (double& c : mcost) {
        c = static_cast<double>(static_cast<long long>(c * SCALE));
    }

    long msz[3] = {
        12,
        2 * n_detection + 1,
        n_arcs
    };

    price_t* track_vec = pyCS2(
        msz,
        mtail.data(),
        mhead.data(),
        mlow.data(),
        macap.data(),
        mcost.data()
    );

    if (track_vec == nullptr) {
        throw std::runtime_error("pyCS2 returned null.");
    }

    const long long L = static_cast<long long>(track_vec[0]);

    std::vector<std::vector<int>> tracks;
    std::vector<price_t> sub;
    sub.reserve(128);

    for (long long i = 1; i <= L; ++i) {
        price_t x = track_vec[i];

        if (x > 0) {
            sub.push_back(x);
        } else {
            std::vector<int> tr;

            for (size_t k = 0; k < sub.size(); k += 2) {
                /*
                 * Python:
                 *   new = [int(x/2) for x in sub_traj[::2]]
                 *   traj.append(np.array(new)-1)
                 */
                int node = static_cast<int>(sub[k] / 2) - 1;

                /*
                 * Python pruning later does:
                 *   track[track < N]
                 * Here we directly filter control nodes.
                 */
                if (node >= 0 && node < N_original) {
                    tr.push_back(node);
                }
            }

            if (!tr.empty()) {
                tracks.push_back(std::move(tr));
            }

            sub.clear();
        }
    }

    pyFreeTrackVec(track_vec);

    return tracks;
}


// ============================================================
// 7. pybind exposed functions
// ============================================================

py::tuple solve_easy_fusion_cpp(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> frames1_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> parents1_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> frames2_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> parents2_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> matches_arr,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> vox_flat_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> vox_offsets_arr,
    int dim,
    double threshold
) {
    if (matches_arr.ndim() != 2 || matches_arr.shape(1) < 2) {
        throw std::runtime_error("matches must have shape (M, >=2).");
    }

    if (vox_flat_arr.ndim() != 2 || vox_flat_arr.shape(1) != dim) {
        throw std::runtime_error("vox_flat shape does not match dim.");
    }

    if (dim != 2 && dim != 3) {
        throw std::runtime_error("Only 2D or 3D voxels are supported.");
    }

    auto frames1 = frames1_arr.unchecked<1>();
    auto parents1 = parents1_arr.unchecked<1>();
    auto frames2 = frames2_arr.unchecked<1>();
    auto parents2 = parents2_arr.unchecked<1>();
    auto matches = matches_arr.unchecked<2>();
    auto vox_flat = vox_flat_arr.unchecked<2>();
    auto vox_offsets = vox_offsets_arr.unchecked<1>();

    const int n1 = static_cast<int>(frames1.shape(0));
    const int n2 = static_cast<int>(frames2.shape(0));
    const int N = n1 + n2;

    if (parents1.shape(0) != n1 || parents2.shape(0) != n2) {
        throw std::runtime_error("frames and parents length mismatch.");
    }

    if (vox_offsets.shape(0) != N + 1) {
        throw std::runtime_error("vox_offsets must have length N + 1.");
    }

    std::vector<std::vector<int>> easy_subgraphs;
    std::vector<std::vector<int>> hard_subgraphs;
    std::vector<std::vector<int>> edges;

    build_subgraphs_internal(
        frames1,
        parents1,
        frames2,
        parents2,
        matches,
        easy_subgraphs,
        hard_subgraphs,
        edges
    );

    std::vector<DetArc> detection_arcs;
    std::vector<TransArc> transition_arcs;

    build_easy_arcs(
        easy_subgraphs,
        edges,
        frames1,
        frames2,
        vox_flat.data(0, 0),
        vox_offsets.data(0),
        dim,
        threshold,
        detection_arcs,
        transition_arcs
    );

    std::vector<std::vector<int>> tracks_easy =
        run_cinda_cpp(detection_arcs, transition_arcs, N);

    return py::make_tuple(
        tracks_easy,
        hard_subgraphs,
        edges
    );
}


py::tuple build_easy_arcs_debug_cpp(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> frames1_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> parents1_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> frames2_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> parents2_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> matches_arr,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> vox_flat_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> vox_offsets_arr,
    int dim,
    double threshold
) {
    if (matches_arr.ndim() != 2 || matches_arr.shape(1) < 2) {
        throw std::runtime_error("matches must have shape (M, >=2).");
    }

    if (vox_flat_arr.ndim() != 2 || vox_flat_arr.shape(1) != dim) {
        throw std::runtime_error("vox_flat shape does not match dim.");
    }

    if (dim != 2 && dim != 3) {
        throw std::runtime_error("Only 2D or 3D voxels are supported.");
    }

    auto frames1 = frames1_arr.unchecked<1>();
    auto parents1 = parents1_arr.unchecked<1>();
    auto frames2 = frames2_arr.unchecked<1>();
    auto parents2 = parents2_arr.unchecked<1>();
    auto matches = matches_arr.unchecked<2>();
    auto vox_flat = vox_flat_arr.unchecked<2>();
    auto vox_offsets = vox_offsets_arr.unchecked<1>();

    const int n1 = static_cast<int>(frames1.shape(0));
    const int n2 = static_cast<int>(frames2.shape(0));
    const int N = n1 + n2;

    if (parents1.shape(0) != n1 || parents2.shape(0) != n2) {
        throw std::runtime_error("frames and parents length mismatch.");
    }

    if (vox_offsets.shape(0) != N + 1) {
        throw std::runtime_error("vox_offsets must have length N + 1.");
    }

    std::vector<std::vector<int>> easy_subgraphs;
    std::vector<std::vector<int>> hard_subgraphs;
    std::vector<std::vector<int>> edges;

    build_subgraphs_internal(
        frames1,
        parents1,
        frames2,
        parents2,
        matches,
        easy_subgraphs,
        hard_subgraphs,
        edges
    );

    std::vector<DetArc> detection_arcs_vec;
    std::vector<TransArc> transition_arcs_vec;

    build_easy_arcs(
        easy_subgraphs,
        edges,
        frames1,
        frames2,
        vox_flat.data(0, 0),
        vox_offsets.data(0),
        dim,
        threshold,
        detection_arcs_vec,
        transition_arcs_vec
    );

    const py::ssize_t det_rows =
        static_cast<py::ssize_t>(detection_arcs_vec.size());

    const py::ssize_t trans_rows =
        static_cast<py::ssize_t>(transition_arcs_vec.size());

    std::vector<py::ssize_t> det_shape = {det_rows, 4};
    std::vector<py::ssize_t> trans_shape = {trans_rows, 3};

    py::array_t<double> detection_arcs(det_shape);
    py::array_t<double> transition_arcs(trans_shape);

    auto det = detection_arcs.mutable_unchecked<2>();

    for (py::ssize_t i = 0; i < det_rows; ++i) {
        const DetArc& a = detection_arcs_vec[static_cast<size_t>(i)];

        det(i, 0) = a.id;
        det(i, 1) = a.c_en;
        det(i, 2) = a.c_ex;
        det(i, 3) = a.c_det;
    }

    auto tr = transition_arcs.mutable_unchecked<2>();

    for (py::ssize_t i = 0; i < trans_rows; ++i) {
        const TransArc& a = transition_arcs_vec[static_cast<size_t>(i)];

        tr(i, 0) = a.src;
        tr(i, 1) = a.dst;
        tr(i, 2) = a.cost;
    }

    return py::make_tuple(
        detection_arcs,
        transition_arcs,
        easy_subgraphs,
        hard_subgraphs,
        edges
    );
}


// ============================================================
// 8. Module binding
// ============================================================

PYBIND11_MODULE(fast_easy_fusion, m) {
    m.doc() = "Fast C++ implementation of INTACT Step 4-1 and Step 4-2 easy-subgraph fusion.";

    m.def(
        "solve_easy_fusion_cpp",
        &solve_easy_fusion_cpp,
        py::arg("frames1"),
        py::arg("parents1"),
        py::arg("frames2"),
        py::arg("parents2"),
        py::arg("matches"),
        py::arg("vox_flat"),
        py::arg("vox_offsets"),
        py::arg("dim"),
        py::arg("threshold")
    );

    m.def(
        "build_easy_arcs_debug_cpp",
        &build_easy_arcs_debug_cpp,
        py::arg("frames1"),
        py::arg("parents1"),
        py::arg("frames2"),
        py::arg("parents2"),
        py::arg("matches"),
        py::arg("vox_flat"),
        py::arg("vox_offsets"),
        py::arg("dim"),
        py::arg("threshold")
    );
}