# HexPlane 함수 호출 순서 가이드

6개 평면 생성부터 최종 feature 생성까지의 전체 흐름을 단계별로 설명합니다.

---

## 📋 전체 흐름 개요

```
[초기화 단계] → [추론 단계]
     ↓              ↓
평면 생성        Feature 추출
```

---

## 🔧 1단계: 초기화 단계 (모델 생성 시)

모델이 처음 생성될 때 한 번만 실행됩니다.

### 1-1. Deformation 클래스 초기화
**파일**: `scene/deformation.py`
```python
Deformation.__init__()
  ↓
self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
```

### 1-2. HexPlaneField 클래스 초기화
**파일**: `scene/hexplane.py`  
**함수**: `HexPlaneField.__init__()`

**순서**:
1. AABB 초기화
2. 설정 저장 (`self.grid_config`, `self.multiscale_res_multipliers`)
3. **각 해상도 레벨마다 반복** (`for res in self.multiscale_res_multipliers`):
   - 해상도 조정
   - **`init_grid_param()` 호출** ← 6개 평면 생성

### 1-3. 6개 평면 생성
**파일**: `scene/hexplane.py`  
**함수**: `init_grid_param()`

**실행 내용**:
```python
# 1. 4D 공간에서 2D 평면 조합 생성
coo_combs = list(itertools.combinations(range(4), 2))
# 결과: [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
#      = [xy, xz, xt, yz, yt, zt]

# 2. 각 평면에 대해 그리드 파라미터 생성
for ci, coo_comb in enumerate(coo_combs):
    # 평면 파라미터 초기화
    new_grid_coef = nn.Parameter(...)
    # 시간 평면이면 1로, 공간 평면이면 uniform 초기화
    grid_coefs.append(new_grid_coef)

# 3. 6개 평면 리스트 반환
return grid_coefs  # [xy, xz, xt, yz, yt, zt]
```

**결과**: 각 해상도 레벨마다 6개 평면이 생성되어 `self.grids`에 저장됩니다.

---

## 🚀 2단계: 추론 단계 (Forward Pass)

점 좌표가 입력될 때마다 실행됩니다.

### 2-1. HexPlaneField Forward 호출
**파일**: `scene/hexplane.py`  
**함수**: `HexPlaneField.forward()`

```python
HexPlaneField.forward(pts, timestamps)
  ↓
HexPlaneField.get_density(pts, timestamps)
```

### 2-2. 점 좌표 정규화 및 4D 좌표 생성
**파일**: `scene/hexplane.py`  
**함수**: `HexPlaneField.get_density()`

**순서**:
1. **`normalize_aabb()` 호출** - 점 좌표를 [-1, 1] 범위로 정규화
2. `torch.cat()` - 공간 좌표(x,y,z)와 시간(t) 결합 → 4D 좌표 생성
3. `reshape()` - 배치 차원 평탄화 → [N, 4] 형태
4. **`interpolate_ms_features()` 호출** ← Feature 추출 시작

### 2-3. 다중해상도 Feature 추출
**파일**: `scene/hexplane.py`  
**함수**: `interpolate_ms_features()`

**순서**:
1. **평면 조합 생성** (6개 평면 인덱스)
   ```python
   coo_combs = list(itertools.combinations(range(4), 2))
   # [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
   ```

2. **각 해상도 레벨마다 반복** (`for scale_id, grid in enumerate(ms_grids)`):
   
   a. **각 평면마다 반복** (`for ci, coo_comb in enumerate(coo_combs)`):
      - **`grid_sample_wrapper()` 호출** ← Bilinear interpolation
      - Feature 추출: `interp_out_plane = [N, feature_dim]`
      - **곱셈 결합**: `interp_space = interp_space * interp_out_plane`
   
   b. 해상도 레벨별 feature 저장: `multi_scale_interp.append(interp_space)`

3. **모든 해상도 레벨 concat**: `torch.cat(multi_scale_interp, dim=-1)`

### 2-4. Bilinear Interpolation (각 평면에서)
**파일**: `scene/hexplane.py`  
**함수**: `grid_sample_wrapper()`

**실행 내용**:
```python
# 1. 좌표 추출: pts[..., coo_comb]
#    예: coo_comb=(0,1)이면 [x, y] 좌표만 추출

# 2. Bilinear interpolation 수행
interp = F.grid_sample(
    grid[ci],      # 평면 그리드 [1, feature_dim, H, W]
    coords,        # 샘플링 좌표 [B, 1, N, 2]
    mode='bilinear'
)

# 3. 결과 형태 변환: [N, feature_dim]
return interp
```

---

## 📊 전체 호출 트리

```
[초기화]
Deformation.__init__()
  └─ HexPlaneField.__init__()
      └─ for res in multires:
          └─ init_grid_param()  ← 6개 평면 생성
              ├─ itertools.combinations()  ← 평면 조합 생성
              └─ for coo_comb in coo_combs:
                  └─ 평면 파라미터 초기화

[추론]
HexPlaneField.forward(pts, timestamps)
  └─ HexPlaneField.get_density(pts, timestamps)
      ├─ normalize_aabb()  ← 좌표 정규화
      ├─ torch.cat()  ← 4D 좌표 생성
      └─ interpolate_ms_features()  ← Feature 추출
          ├─ itertools.combinations()  ← 평면 조합 생성
          └─ for scale_id, grid in ms_grids:  ← 각 해상도 레벨
              └─ for ci, coo_comb in coo_combs:  ← 각 평면
                  └─ grid_sample_wrapper()  ← Bilinear interpolation
                      └─ F.grid_sample()  ← 실제 interpolation
          └─ torch.cat()  ← 최종 feature concat
```

---

## 🔍 상세 함수 호출 순서

### 초기화 단계
1. `Deformation.__init__()` (deformation.py:26)
2. `HexPlaneField.__init__()` (hexplane.py:301)
3. `init_grid_param()` (hexplane.py:368) - 각 해상도마다 호출
   - `itertools.combinations()` (hexplane.py:137) - 6개 평면 조합 생성
   - 평면 파라미터 초기화 (hexplane.py:144-166)

### 추론 단계
1. `HexPlaneField.forward()` (hexplane.py:474)
2. `HexPlaneField.get_density()` (hexplane.py:412)
3. `normalize_aabb()` (hexplane.py:435)
4. `interpolate_ms_features()` (hexplane.py:458)
   - `itertools.combinations()` (hexplane.py:208) - 평면 조합 생성
   - 각 해상도 레벨마다:
     - 각 평면마다:
       - `grid_sample_wrapper()` (hexplane.py:251)
         - `F.grid_sample()` (hexplane.py:79) - Bilinear interpolation
5. `torch.cat()` (hexplane.py:286) - 최종 feature concat

---

## 💡 핵심 포인트

1. **초기화**: `init_grid_param()`이 각 해상도 레벨마다 6개 평면을 생성
2. **추론**: `interpolate_ms_features()`가 각 평면에서 feature를 추출하고 결합
3. **평면 조합**: `itertools.combinations(range(4), 2)`로 6개 평면 인덱스 생성
4. **Feature 결합**: 
   - 같은 해상도 내: 6개 평면 feature를 **곱셈**으로 결합
   - 다른 해상도 간: 여러 해상도 feature를 **concat**으로 결합

---

## 📝 코드 위치 요약

| 단계 | 함수명 | 파일 | 라인 |
|------|--------|------|------|
| 초기화 시작 | `Deformation.__init__` | deformation.py | 26 |
| HexPlane 초기화 | `HexPlaneField.__init__` | hexplane.py | 301 |
| 평면 생성 | `init_grid_param` | hexplane.py | 93 |
| Forward 시작 | `HexPlaneField.forward` | hexplane.py | 474 |
| 좌표 정규화 | `normalize_aabb` | hexplane.py | 19 |
| Feature 추출 | `interpolate_ms_features` | hexplane.py | 177 |
| Interpolation | `grid_sample_wrapper` | hexplane.py | 32 |

