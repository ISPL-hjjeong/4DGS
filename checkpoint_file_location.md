# 체크포인트 파일 저장 위치

## 🎯 핵심 답변

체크포인트 파일은 다음 경로에 저장됩니다:

```
./output/{expname}/chkpnt_{stage}_{iteration}.pth
```

---

## 📁 저장 경로 상세

### 1. 기본 경로 결정

```python
# train.py의 prepare_output_and_logger() 함수
def prepare_output_and_logger(expname):    
    if not args.model_path:
        unique_str = expname  # expname 인자 사용
        args.model_path = os.path.join("./output/", unique_str)
    
    # 출력 폴더 생성
    os.makedirs(args.model_path, exist_ok=True)
    print("Output folder: {}".format(args.model_path))
```

**결과**: `./output/{expname}/` 디렉토리에 저장

---

### 2. 체크포인트 파일 저장

```python
# train.py의 scene_reconstruction() 함수
if (iteration in checkpoint_iterations):
    print("\n[ITER {}] Saving Checkpoint".format(iteration))
    torch.save(
        (gaussians.capture(), iteration), 
        scene.model_path + "/chkpnt" + f"_{stage}_" + str(iteration) + ".pth"
    )
```

**파일명 형식**: `chkpnt_{stage}_{iteration}.pth`

---

## 📂 실제 파일 구조 예시

### 예시 1: expname="dnerf/bouncingballs"

```
./output/
└── dnerf/
    └── bouncingballs/
        ├── chkpnt_coarse_200.pth          # Coarse stage 체크포인트
        ├── chkpnt_coarse_3000.pth
        ├── chkpnt_fine_14000.pth           # Fine stage 체크포인트
        ├── chkpnt_fine_30000.pth
        ├── point_cloud/                    # Point cloud 저장 위치
        │   ├── coarse_iteration_3000/
        │   │   ├── point_cloud.ply
        │   │   ├── deformation.pth
        │   │   ├── deformation_table.pth
        │   │   └── deformation_accum.pth
        │   └── iteration_14000/
        │       ├── point_cloud.ply
        │       ├── deformation.pth
        │       ├── deformation_table.pth
        │       └── deformation_accum.pth
        └── cfg_args                        # 설정 파일
```

---

## 🔍 파일 저장 위치 상세

### 1. Checkpoint 파일 (전체 모델 저장)

**경로**: `./output/{expname}/chkpnt_{stage}_{iteration}.pth`

**내용**:
- `gaussians.capture()`의 반환값
  - `active_sh_degree`
  - `_xyz` (Gaussian 위치)
  - `_deformation.state_dict()` ← **HexPlane feature map 포함!**
  - `_deformation_table`
  - `_features_dc`, `_features_rest` (Gaussian 색상)
  - `_scaling`, `_rotation`, `_opacity` (Gaussian 속성)
  - `optimizer.state_dict()`
  - 기타 학습 상태

**예시 파일명**:
- `chkpnt_coarse_3000.pth`
- `chkpnt_fine_14000.pth`

---

### 2. Point Cloud 및 Deformation 파일

**경로**: `./output/{expname}/point_cloud/{stage}_iteration_{iteration}/`

**저장 함수**:
```python
# scene/__init__.py
def save(self, iteration, stage):
    if stage == "coarse":
        point_cloud_path = os.path.join(
            self.model_path, 
            "point_cloud/coarse_iteration_{}".format(iteration)
        )
    else:
        point_cloud_path = os.path.join(
            self.model_path, 
            "point_cloud/iteration_{}".format(iteration)
        )
    
    self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    self.gaussians.save_deformation(point_cloud_path)
```

**저장되는 파일**:
- `point_cloud.ply`: Gaussian point cloud
- `deformation.pth`: Deformation 네트워크 파라미터 (HexPlane 포함)
- `deformation_table.pth`: Deformation 테이블
- `deformation_accum.pth`: Deformation 누적값

---

## 💾 HexPlane Feature Map 저장 위치

### Checkpoint 파일 내부 구조

```python
# chkpnt_coarse_3000.pth 파일 내용
(
    (
        active_sh_degree,
        _xyz,
        {
            # Deformation 모듈의 state_dict
            'grid.grids.0.0': tensor([1, 32, 64, 64]),    # 해상도 1, xy 평면
            'grid.grids.0.1': tensor([1, 32, 64, 64]),    # 해상도 1, xz 평면
            'grid.grids.0.2': tensor([1, 32, 25, 64]),    # 해상도 1, xt 평면
            'grid.grids.0.3': tensor([1, 32, 64, 64]),    # 해상도 1, yz 평면
            'grid.grids.0.4': tensor([1, 32, 25, 64]),    # 해상도 1, yt 평면
            'grid.grids.0.5': tensor([1, 32, 25, 64]),    # 해상도 1, zt 평면
            'grid.grids.1.0': tensor([1, 32, 128, 128]),  # 해상도 2, xy 평면
            # ... (모든 해상도, 모든 평면)
            'grid.aabb': tensor([[1.6, 1.6, 1.6], [-1.6, -1.6, -1.6]]),
            'feature_out.0.weight': tensor([...]),
            # ... (MLP 파라미터)
        },
        _deformation_table,
        _features_dc,
        _features_rest,
        _scaling,
        _rotation,
        _opacity,
        max_radii2D,
        xyz_gradient_accum,
        denom,
        optimizer.state_dict(),
        spatial_lr_scale,
    ),
    iteration  # 3000
)
```

---

## 📝 실제 사용 예시

### 학습 시 저장

```bash
# 학습 실행
python train.py \
    -s data/dnerf/bouncingballs \
    --expname "dnerf/bouncingballs" \
    --checkpoint_iterations 200 3000 14000

# 저장되는 파일:
# ./output/dnerf/bouncingballs/chkpnt_coarse_200.pth
# ./output/dnerf/bouncingballs/chkpnt_coarse_3000.pth
# ./output/dnerf/bouncingballs/chkpnt_fine_14000.pth
```

### 체크포인트 로드

```bash
# 체크포인트에서 학습 재개
python train.py \
    -s data/dnerf/bouncingballs \
    --expname "dnerf/bouncingballs" \
    --start_checkpoint "output/dnerf/bouncingballs/chkpnt_coarse_200.pth"
```

---

## 🔍 파일 크기 예시

### Checkpoint 파일 크기 (multires=[1,2,4,8])

- **HexPlane 파라미터**: 약 136MB (35,733,256개 파라미터 × 4 bytes)
- **Gaussian 파라미터**: 데이터셋에 따라 다름 (수만~수백만 개)
- **Optimizer 상태**: 약 2배 크기 (momentum 등 포함)
- **전체 파일**: 약 200MB ~ 수GB (데이터셋 크기에 따라)

---

## 📂 요약

1. **Checkpoint 파일 경로**: 
   - `./output/{expname}/chkpnt_{stage}_{iteration}.pth`
   - 예: `./output/dnerf/bouncingballs/chkpnt_coarse_3000.pth`

2. **Deformation 파일 경로**:
   - `./output/{expname}/point_cloud/{stage}_iteration_{iteration}/deformation.pth`
   - 예: `./output/dnerf/bouncingballs/point_cloud/coarse_iteration_3000/deformation.pth`

3. **HexPlane Feature Map 저장 위치**:
   - Checkpoint 파일 내부: `_deformation.state_dict()['grid.grids.*.*']`
   - Deformation 파일: `deformation.pth` (별도 저장)

4. **기본 경로**:
   - `--expname` 인자로 지정하거나, 없으면 `./output/` 디렉토리 사용

---

## 🎯 핵심 포인트

- **체크포인트 파일**: 전체 모델 상태 저장 (HexPlane 포함)
- **Deformation 파일**: Deformation 네트워크만 별도 저장
- **모든 HexPlane feature map**: 두 파일 모두에 포함됨

**실제 경로**: `./output/{expname}/chkpnt_{stage}_{iteration}.pth` 🎯

