# DNeRF 데이터셋 초기 Point Cloud 추출 파이프라인

이 문서는 dnerf 데이터셋에서 초기 3D Gaussian을 생성하기 위한 point cloud 추출 과정을 코드 기반으로 설명합니다.

## 전체 파이프라인 개요

```
transforms_train.json 
    ↓
[1단계] scripts/blender2colmap.py - 프레임 선택 및 COLMAP 형식 변환
    ↓
[2단계] colmap.sh - COLMAP 파이프라인 실행
    ↓
[3단계] fused.ply 생성 (Dense Reconstruction)
    ↓
[4단계] scene/dataset_readers.py - Point Cloud 로딩
    ↓
[5단계] scene/gaussian_model.py - 3D Gaussian 초기화
```

---

## 1단계: 프레임 선택 및 COLMAP 형식 변환

**파일**: `scripts/blender2colmap.py`

### 1.1 프레임 선택 로직

```python
# 58-66번 줄
idx=0
sizes=1
cnt=0
# 프레임 수를 약 200개 이하로 유지하도록 간격 계산
while len(meta['frames'])//sizes > 200:
    sizes += 1

for frame in meta['frames']:
    cnt+=1
    # sizes 간격마다 프레임 선택
    if cnt % sizes != 0:
        continue  # 이 프레임은 건너뛰기
```

**설명**:
- `transforms_train.json`에서 모든 프레임을 읽어옴
- 전체 프레임 수를 200개 이하로 맞추기 위해 `sizes` 간격을 자동 계산
- 예: 1000개 프레임 → `sizes=5` → 5프레임마다 선택 → 약 200개 프레임

### 1.2 카메라 파라미터 추출

```python
# 40-52번 줄
with open(camera_json) as f:
    meta = json.load(f)
    
# 이미지 크기와 focal length 추출
try:
    image_size = meta['w'], meta['h']
    focal = [meta['fl_x'], meta['fl_y']]
except:
    # fallback 로직들...
```

### 1.3 COLMAP 형식 파일 생성

```python
# 67-80번 줄
for frame in meta['frames']:
    # Transform matrix를 COLMAP 형식으로 변환
    matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
    R = -np.transpose(matrix[:3,:3])  # Rotation
    T = -matrix[:3, 3]                 # Translation
    
    # Quaternion으로 변환
    qevc = rotmat2qvec(np.transpose(R))
    
    # COLMAP images.txt 형식으로 저장
    print(idx+1, qevc, T, 1, frame['file_path']+".png", 
          file=object_images_file)
    
    # COLMAP cameras.txt 형식으로 저장
    print(idx, "SIMPLE_PINHOLE", image_size[0], image_size[1], 
          focal[0], image_size[0]/2, image_size[1]/2, 
          file=object_cameras_file)
    
    # 이미지 파일 복사
    shutil.copy(source_image, destination_image)
```

**출력 파일**:
- `{dataset_dir}/sparse_/images.txt` - 카메라 포즈 정보
- `{dataset_dir}/sparse_/cameras.txt` - 카메라 내부 파라미터
- `{dataset_dir}/image_colmap/` - 선택된 프레임 이미지들

---

## 2단계: COLMAP 파이프라인 실행

**파일**: `colmap.sh`

### 2.1 Feature Extraction (특징점 추출)

```bash
# 17번 줄
colmap feature_extractor \
    --database_path $workdir/colmap/database.db \
    --image_path $workdir/colmap/images \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model SIMPLE_PINHOLE \
    --SiftExtraction.max_image_size 4096 \
    --SiftExtraction.max_num_features 16384 \
    --SiftExtraction.domain_size_pooling 1 \
    --SiftExtraction.estimate_affine_shape 1
```

**설명**:
- 각 이미지에서 SIFT 특징점 추출
- 최대 16384개 특징점, 이미지 크기 최대 4096px

### 2.2 Camera Database 업데이트

```bash
# 18번 줄
python database.py \
    --database_path $workdir/colmap/database.db \
    --txt_path $workdir/colmap/sparse_custom/cameras.txt
```

**설명** (`database.py`):
- COLMAP 데이터베이스에 카메라 파라미터 업데이트
- `cameras.txt`에서 카메라 정보를 읽어 SQLite 데이터베이스에 저장

### 2.3 Feature Matching (특징점 매칭)

```bash
# 19번 줄
colmap exhaustive_matcher \
    --database_path $workdir/colmap/database.db \
    --SiftMatching.use_gpu 0
```

**설명**:
- 모든 이미지 쌍에 대해 특징점 매칭 수행
- GPU 사용 안 함 (CPU로 실행)

### 2.4 Sparse Reconstruction (희소 재구성)

```bash
# 21번 줄
colmap point_triangulator \
    --database_path $workdir/colmap/database.db \
    --image_path $workdir/colmap/images \
    --input_path $workdir/colmap/sparse_custom \
    --output_path $workdir/colmap/sparse/0 \
    --clear_points 1
```

**설명**:
- 이미지 간 매칭된 특징점을 3D 포인트로 삼각측량(triangulation)
- 희소한 3D 포인트 클라우드 생성
- 카메라 포즈 최적화

### 2.5 Dense Reconstruction (밀집 재구성)

```bash
# 24번 줄: Image Undistortion
colmap image_undistorter \
    --image_path $workdir/colmap/images \
    --input_path $workdir/colmap/sparse/0 \
    --output_path $workdir/colmap/dense/workspace

# 25번 줄: Patch Match Stereo
colmap patch_match_stereo \
    --workspace_path $workdir/colmap/dense/workspace

# 26번 줄: Stereo Fusion
colmap stereo_fusion \
    --workspace_path $workdir/colmap/dense/workspace \
    --output_path $workdir/colmap/dense/workspace/fused.ply
```

**설명**:
- **Image Undistortion**: 렌즈 왜곡 보정
- **Patch Match Stereo**: 다중 뷰에서 깊이 맵 생성
- **Stereo Fusion**: 깊이 맵들을 융합하여 최종 point cloud 생성

**출력 파일**:
- `{dataset_dir}/colmap/dense/workspace/fused.ply` - 최종 밀집 point cloud

---

## 3단계: Point Cloud 로딩

**파일**: `scene/dataset_readers.py`

### 3.1 데이터셋 타입 감지

```python
# scene/__init__.py 48-50번 줄
elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
    print("Found transforms_train.json file, assuming Blender data set!")
    scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, ...)
    dataset_type="blender"
```

**설명**:
- `transforms_train.json` 파일 존재 여부로 dnerf/Blender 데이터셋 감지
- `readNerfSyntheticInfo` 함수 호출

### 3.2 Point Cloud 파일 경로 확인

```python
# scene/dataset_readers.py 327-341번 줄
ply_path = os.path.join(path, "fused.ply")

if not os.path.exists(ply_path):
    # Point cloud가 없으면 랜덤 포인트 생성
    num_pts = 2000
    print(f"Generating random point cloud ({num_pts})...")
    
    # Blender 씬 경계 내에서 랜덤 포인트 생성
    xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), 
                          normals=np.zeros((num_pts, 3)))
else:
    # PLY 파일에서 point cloud 로딩
    pcd = fetchPly(ply_path)
```

**설명**:
- `fused.ply` 파일이 있으면 로딩
- 없으면 랜덤 포인트 2000개 생성 (fallback)

### 3.3 PLY 파일 파싱

```python
# scene/dataset_readers.py 124-130번 줄
def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    
    # 위치 (x, y, z)
    positions = np.vstack([vertices['x'], 
                           vertices['y'], 
                           vertices['z']]).T
    
    # 색상 (red, green, blue) - 0~255 → 0~1로 정규화
    colors = np.vstack([vertices['red'], 
                       vertices['green'], 
                       vertices['blue']]).T / 255.0
    
    # 법선 벡터 (nx, ny, nz)
    normals = np.vstack([vertices['nx'], 
                        vertices['ny'], 
                        vertices['nz']]).T
    
    return BasicPointCloud(points=positions, 
                          colors=colors, 
                          normals=normals)
```

**설명**:
- PLY 파일에서 vertex 데이터 읽기
- 위치, 색상, 법선 벡터 추출
- `BasicPointCloud` 객체로 반환

---

## 4단계: 3D Gaussian 초기화

**파일**: `scene/gaussian_model.py`, `scene/__init__.py`

### 4.1 Point Cloud를 Gaussian으로 변환

```python
# scene/__init__.py 93-94번 줄
else:
    self.gaussians.create_from_pcd(scene_info.point_cloud, 
                                   self.cameras_extent, 
                                   self.maxtime)
```

### 4.2 Gaussian 파라미터 초기화

```python
# scene/gaussian_model.py 138-165번 줄
def create_from_pcd(self, pcd: BasicPointCloud, 
                    spatial_lr_scale: float, time_line: int):
    
    # 1. 위치 (xyz)
    fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
    
    # 2. 색상 → Spherical Harmonics 변환
    fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
    features = torch.zeros((fused_color.shape[0], 3, 
                           (self.max_sh_degree + 1) ** 2)).float().cuda()
    features[:, :3, 0] = fused_color  # DC component만 사용
    
    # 3. 스케일 (크기) - 최근접 이웃 거리 기반
    dist2 = torch.clamp_min(
        distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 
        0.0000001
    )
    scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
    
    # 4. 회전 - 초기값은 단위 쿼터니언
    rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    rots[:, 0] = 1  # w=1, x=y=z=0 (회전 없음)
    
    # 5. 불투명도 - 초기값 0.1
    opacities = inverse_sigmoid(
        0.1 * torch.ones((fused_point_cloud.shape[0], 1), 
                        dtype=torch.float, device="cuda")
    )
    
    # 6. 파라미터로 등록 (학습 가능하게)
    self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2)
                                     .contiguous().requires_grad_(True))
    self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2)
                                       .contiguous().requires_grad_(True))
    self._scaling = nn.Parameter(scales.requires_grad_(True))
    self._rotation = nn.Parameter(rots.requires_grad_(True))
    self._opacity = nn.Parameter(opacities.requires_grad_(True))
```

**주요 초기화 값**:
- **위치 (xyz)**: Point cloud의 3D 좌표 그대로 사용
- **색상**: RGB → Spherical Harmonics 변환 (DC component만)
- **스케일**: 각 포인트의 최근접 이웃 거리의 로그값
- **회전**: 단위 쿼터니언 (회전 없음)
- **불투명도**: 0.1 (sigmoid 역변환 적용)

---

## 요약

1. **프레임 선택**: `transforms_train.json`에서 약 200개 프레임 자동 선택
2. **COLMAP 변환**: 선택된 프레임을 COLMAP 형식으로 변환
3. **특징점 추출/매칭**: SIFT 특징점으로 이미지 간 매칭
4. **삼각측량**: 매칭된 특징점을 3D 포인트로 변환
5. **밀집 재구성**: 다중 뷰 스테레오로 밀집 point cloud 생성 (`fused.ply`)
6. **로딩**: PLY 파일에서 point cloud 읽기
7. **Gaussian 초기화**: Point cloud를 3D Gaussian 파라미터로 변환

**최종 결과**: 초기 3D Gaussian 모델이 생성되어 학습이 시작됩니다.

