import math
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# ============================================================
# 1. Gerar um poligono irregular que imita um "lago"
# ============================================================

def generate_lake_polygon(n_boundary=80, seed=1):
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * math.pi, n_boundary, endpoint=False)
    # raio varia um pouco -> borda irregular, com concavidades leves
    radii = 0.8 + 0.4 * rng.random(n_boundary)
    xs = radii * np.cos(angles)
    ys = 0.7 * radii * np.sin(angles)  # um pouco achatado
    poly = np.stack([xs, ys], axis=1)

    # translada para coordenadas positivas e normaliza para ~[0,1]^2
    min_xy = poly.min(axis=0)
    poly -= min_xy
    max_xy = poly.max(axis=0)
    scale = max(max_xy[0], max_xy[1])
    poly /= scale
    return poly  # array (m,2)

# ============================================================
# 2. Pontos dentro do poligono (teste ponto-no-poligono)
# ============================================================

def point_in_polygon(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            t = (y - y1) / (y2 - y1 + 1e-18)
            x_cross = x1 + t * (x2 - x1)
            if x < x_cross:
                inside = not inside
    return inside

def sample_points_in_polygon(poly, n_points, seed=1):
    rng = np.random.default_rng(seed)
    xs_min, ys_min = poly.min(axis=0)
    xs_max, ys_max = poly.max(axis=0)
    pts = []
    while len(pts) < n_points:
        x = rng.random() * (xs_max - xs_min) + xs_min
        y = rng.random() * (ys_max - ys_min) + ys_min
        if point_in_polygon((x, y), poly):
            pts.append((x, y))
    return np.array(pts)

# ============================================================
# 3. Delaunay via SciPy + utilidades geometricas
# ============================================================

def delaunay_triangulation(points):
    pts = np.asarray(points, float)
    tri = Delaunay(pts)
    return tri.simplices.tolist()   # lista de triangulos (i,j,k)

def orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

def segments_intersect(p1, p2, p3, p4):
    def on_segment(a, b, c):
        return (min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12 and
                min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12)

    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)

    if (o1 == 0 and on_segment(p1, p2, p3)) or \
       (o2 == 0 and on_segment(p1, p2, p4)) or \
       (o3 == 0 and on_segment(p3, p4, p1)) or \
       (o4 == 0 and on_segment(p3, p4, p2)):
        return True

    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

def circumcircle(p1, p2, p3):
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    ax2ay2 = ax * ax + ay * ay
    bx2by2 = bx * bx + by * by
    cx2cy2 = cx * cx + cy * cy
    ux = (ax2ay2 * (by - cy) +
          bx2by2 * (cy - ay) +
          cx2cy2 * (ay - by)) / d
    uy = (ax2ay2 * (cx - bx) +
          bx2by2 * (ax - cx) +
          cx2cy2 * (bx - ax)) / d
    r2 = (ux - ax) ** 2 + (uy - ay) ** 2
    return ux, uy, r2

def triangle_geom(pts, tri):
    a, b, c = tri
    p, q, r = pts[a], pts[b], pts[c]

    ab = np.linalg.norm(q - p)
    bc = np.linalg.norm(r - q)
    ca = np.linalg.norm(p - r)

    area = abs(orient(p, q, r)) / 2.0

    def angle(opposite, side1, side2):
        denom = 2.0 * side1 * side2
        if denom == 0.0:
            return 0.0
        cosang = (side1 * side1 + side2 * side2 - opposite * opposite) / denom
        cosang = max(-1.0, min(1.0, cosang))
        return math.degrees(math.acos(cosang))

    angA = angle(bc, ab, ca)
    angB = angle(ca, ab, bc)
    angC = angle(ab, bc, ca)
    return area, min(angA, angB, angC), max(ab, bc, ca)

def triangle_circumcenter(pts, tri):
    a, b, c = tri
    res = circumcircle(pts[a], pts[b], pts[c])
    if res is None:
        return (pts[a] + pts[b] + pts[c]) / 3.0
    cx, cy, _ = res
    return np.array([cx, cy])

# ============================================================
# 4. Recuperacao de restricoes (CDT leve)
# ============================================================

def build_edge_to_tris(tris):
    edge_to_tris = {}
    for idx, t in enumerate(tris):
        a, b, c = t
        for e in [(a, b), (b, c), (c, a)]:
            key = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
            edge_to_tris.setdefault(key, []).append(idx)
    return edge_to_tris

def segment_in_triangulation(seg, tris):
    a, b = seg
    for t in tris:
        s = set(t)
        if a in s and b in s:
            return True
    return False

def recover_constraints(points, triangles, segments, max_iters=200):
    pts = np.asarray(points)
    tris = triangles[:]

    for (a, b) in segments:
        if a == b:
            continue
        target = (a, b)
        it = 0
        while not segment_in_triangulation(target, tris):
            it += 1
            if it > max_iters:
                print("Aviso: limite ao recuperar segmento", target)
                break

            found = False
            edge_map = build_edge_to_tris(tris)

            for (i, j), tri_ids in edge_map.items():
                if i in target or j in target:
                    continue
                if len(tri_ids) != 2:
                    continue
                if not segments_intersect(pts[i], pts[j], pts[a], pts[b]):
                    continue

                t1 = tris[tri_ids[0]]
                t2 = tris[tri_ids[1]]

                opp1 = [v for v in t1 if v not in (i, j)][0]
                opp2 = [v for v in t2 if v not in (i, j)][0]

                def tri_eq(u, v):
                    return set(u) == set(v)

                tris = [t for t in tris
                        if not (tri_eq(t, t1) or tri_eq(t, t2))]
                tris.append((opp1, opp2, i))
                tris.append((opp1, opp2, j))

                found = True
                break

            if not found:
                print("Nao foi possivel recuperar", target)
                break

    return tris

# ============================================================
# 5. Selecionar triangulos "ruins" + encroachment
# ============================================================

def get_bad_triangles(pts, tris, min_angle_deg, poly, max_bad=None):
    bad = []
    for idx, tri in enumerate(tris):
        a, b, c = tri
        pa, pb, pc = pts[a], pts[b], pts[c]
        cx = (pa[0] + pb[0] + pc[0]) / 3.0
        cy = (pa[1] + pb[1] + pc[1]) / 3.0
        if not point_in_polygon((cx, cy), poly):
            continue
        area, min_ang, _ = triangle_geom(pts, tri)
        if area < 1e-8:
            continue
        if min_ang < min_angle_deg:
            bad.append((min_ang, idx))
    bad.sort(key=lambda t: t[0])
    indices = [idx for _, idx in bad]
    if max_bad is not None:
        indices = indices[:max_bad]
    return indices

def find_encroached_segment(candidate_point, pts, segments):
    for (a, b) in segments:
        pa, pb = pts[a], pts[b]
        mid = 0.5 * (pa + pb)
        r2 = np.sum((pa - mid) ** 2)
        d2 = np.sum((candidate_point - mid) ** 2)
        if d2 < r2 - 1e-10:
            return (a, b)
    return None

# ============================================================
# 6. Refinamento tipo Chew e Ruppert (poucas iteracoes)
# ============================================================

def chew_refinement(points, segments, poly,
                    min_angle_deg=28.0, iterations=3, max_new_each=40):
    pts = np.array(points, float)
    segs = list(segments)

    for _ in range(iterations):
        tris = delaunay_triangulation(pts)
        tris = recover_constraints(pts, tris, segs)
        bad_idx = get_bad_triangles(pts, tris, min_angle_deg,
                                    poly, max_bad=max_new_each)
        if not bad_idx:
            break

        new_pts = []
        for idx in bad_idx:
            cc = triangle_circumcenter(pts, tris[idx])
            if point_in_polygon(cc, poly):
                new_pts.append(cc)

        if not new_pts:
            break

        pts = np.vstack([pts, np.array(new_pts)])

    tris = delaunay_triangulation(pts)
    tris = recover_constraints(pts, tris, segs)
    return pts, tris

def ruppert_refinement(points, segments, poly,
                       min_angle_deg=28.0, iterations=3, max_new_each=40):
    pts = np.array(points, float)
    segs = list(segments)

    for _ in range(iterations):
        tris = delaunay_triangulation(pts)
        tris = recover_constraints(pts, tris, segs)
        bad_idx = get_bad_triangles(pts, tris, min_angle_deg,
                                    poly, max_bad=max_new_each)
        if not bad_idx:
            break

        new_pts = []
        new_segs = segs
        changed = False

        for idx in bad_idx:
            cc = triangle_circumcenter(pts, tris[idx])
            if not point_in_polygon(cc, poly):
                continue
            enc = find_encroached_segment(cc, pts, segs)
            if enc is not None:
                a, b = enc
                mid = 0.5 * (pts[a] + pts[b])
                if not point_in_polygon(mid, poly):
                    continue
                new_index = len(pts) + len(new_pts)
                new_pts.append(mid)
                updated = []
                for s in new_segs:
                    if s == enc or s == (enc[1], enc[0]):
                        updated.append((a, new_index))
                        updated.append((new_index, b))
                    else:
                        updated.append(s)
                new_segs = updated
                changed = True
            else:
                new_pts.append(cc)

        if not new_pts:
            break

        pts = np.vstack([pts, np.array(new_pts)])
        if changed:
            segs = new_segs

    tris = delaunay_triangulation(pts)
    tris = recover_constraints(pts, tris, segs)
    return pts, tris, segs

# ============================================================
# 7. Plot
# ============================================================

def plot_triangulation(points, triangles, segments=None,
                       poly=None, filename="mesh.png", title=None):
    pts = np.asarray(points)
    tris = np.array(triangles, dtype=int)

    # desenhar apenas triangulos cujo centroide esta dentro do lago
    if poly is not None:
        mask = []
        for tri in tris:
            pa, pb, pc = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            cx = (pa[0] + pb[0] + pc[0]) / 3.0
            cy = (pa[1] + pb[1] + pc[1]) / 3.0
            mask.append(point_in_polygon((cx, cy), poly))
        tris = tris[np.array(mask, dtype=bool)]

    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.triplot(triang, linewidth=0.4)

    # fronteira / restricoes em preto
    if segments is not None:
        for (i, j) in segments:
            pi, pj = pts[i], pts[j]
            ax.plot([pi[0], pj[0]], [pi[1], pj[1]], "k-", linewidth=0.8)

    # desenha contorno do lago
    if poly is not None:
        poly_closed = np.vstack([poly, poly[0]])
        ax.plot(poly_closed[:, 0], poly_closed[:, 1], "k-", linewidth=1.2)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    plt.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)

# ============================================================
# 8. Gera as figuras (evolucao)
# ============================================================

def generate_all():
    # poligono do "lago" + pontos interiores
    poly = generate_lake_polygon(n_boundary=80, seed=1)
    inner = sample_points_in_polygon(poly, n_points=40, seed=1)

    verts = poly
    points = np.vstack([verts, inner])

    # segmentos de fronteira do lago
    m = len(verts)
    segments = [(i, (i + 1) % m) for i in range(m)]

    # 1) Delaunay sem restricao
    tris1 = delaunay_triangulation(points)
    plot_triangulation(points, tris1, segments=None, poly=None,
                       filename="lake_delaunay_sem_restricao.png",
                       title="Delaunay sem restricao")

    check_delaunay(points, tris1)

    # 2) Delaunay com restricoes (CDT aproximada)
    tris2 = recover_constraints(points, tris1, segments)
    plot_triangulation(points, tris2, segments=segments, poly=poly,
                       filename="lake_delaunay_com_restricao.png",
                       title="Delaunay com restricoes")

    # 3) Refinamento tipo Chew
    pts_chew, tris_chew = chew_refinement(
        points, segments, poly,
        min_angle_deg=30.0,
        iterations=5,
        max_new_each=60,
    )
    plot_triangulation(pts_chew, tris_chew, segments=segments, poly=poly,
                       filename="lake_refinamento_chew.png",
                       title="Refinamento tipo Chew")

    # 4) Refinamento tipo Ruppert
    pts_rup, tris_rup, segs_rup = ruppert_refinement(
        points, segments, poly,
        min_angle_deg=30.0,
        iterations=5,
        max_new_each=60,
    )
    plot_triangulation(pts_rup, tris_rup, segments=segs_rup, poly=poly,
                       filename="lake_refinamento_ruppert.png",
                       title="Refinamento tipo Ruppert")

# ============================================================
# 9. Teste
# ============================================================

def check_delaunay(points, triangles, tol=1e-10):
    pts = np.asarray(points, float)
    ok = True
    for t_idx, (i, j, k) in enumerate(triangles):
        p1, p2, p3 = pts[i], pts[j], pts[k]
        cc = circumcircle(p1, p2, p3)
        if cc is None:
            # triângulo quase degenerado; ignora
            continue
        cx, cy, r2 = cc
        for p_idx, p in enumerate(pts):
            if p_idx in (i, j, k):
                continue
            dx = p[0] - cx
            dy = p[1] - cy
            d2 = dx * dx + dy * dy
            if d2 < r2 - tol:  # ponto estritamente dentro do círculo
                print(f"Violacao em triangulo {t_idx}, ponto {p_idx}")
                ok = False
                break
    if ok:
        print("Triangulacao satisfaz o criterio de Delaunay (dentro do tol).")
    else:
        print("Ha violacoes do criterio de Delaunay.")
    return ok

if __name__ == "__main__":
    generate_all()
    print("Imagens geradas:")
    print(" - lake_delaunay_sem_restricao.png")
    print(" - lake_delaunay_com_restricao.png")
    print(" - lake_refinamento_chew.png")
    print(" - lake_refinamento_ruppert.png")
