# Train.py의 Densification 프로세스 상세 설명

이 문서는 `train.py`의 233-290번 줄에 있는 densification 수행 부분에 대한 상세한 설명입니다.

## 개요

`train.py`의 `scene_reconstruction` 함수 내에서, 각 iteration마다 다음 순서로 densification이 수행됩니다:

1. **Loss 계산 및 Backward** (224-230번 줄)
2. **Progress Bar 업데이트** (233-244번 줄)
3. **로깅 및 저장** (246-261번 줄)
4. **Densification 수행** (263-290번 줄) ← **이 부분이 핵심**

---

## 전체 코드 흐름 (233-290번 줄)

```python
with torch.no_grad():  # Gradient 계산 불필요한 부분
    # 1. Progress Bar 업데이트 (233-244번 줄)
    # 2. 로깅 및 저장 (246-261번 줄)
    # 3. Densification (263-290번 줄)
```

### 왜 `torch.no_grad()`를 사용하는가?

- Densification은 **모델 구조를 변경**하는 작업입니다 (Gaussian 추가/제거)
- 이 과정에서는 gradient 계산이 필요 없습니다
- 메모리 효율성과 속도 향상을 위해 `no_grad()` 컨텍스트 사용

---

## 1단계: Progress Bar 및 로깅 (233-261번 줄)

### 1.1 Progress Bar 업데이트 (233-244번 줄)

```python
ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
total_point = gaussians._xyz.shape[0]
if iteration % 10 == 0:
    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                              "psnr": f"{psnr_:.{2}f}",
                              "point":f"{total_point}"})
    progress_bar.update(10)
```

**설명**:
- **EMA (Exponential Moving Average)**: Loss와 PSNR의 지수 이동 평균 계산
  - 최신 값에 40% 가중치, 이전 평균에 60% 가중치
  - 부드러운 추적을 위해 사용
- **10 iteration마다 업데이트**: 너무 자주 업데이트하면 성능 저하
- **total_point**: 현재 Gaussian 개수 표시

### 1.2 로깅 및 저장 (246-261번 줄)

```python
timer.pause()
training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage, scene.dataset_type)
if (iteration in saving_iterations):
    print("\n[ITER {}] Saving Gaussians".format(iteration))
    scene.save(iteration, stage)
if dataset.render_process:
    if (iteration < 1000 and iteration % 10 == 9) \
        or (iteration < 3000 and iteration % 50 == 49) \
            or (iteration < 60000 and iteration %  100 == 99) :
        render_training_image(scene, gaussians, [test_cams[iteration%len(test_cams)]], render, pipe, background, stage+"test", iteration,timer.get_elapsed_time(),scene.dataset_type)
        render_training_image(scene, gaussians, [train_cams[iteration%len(train_cams)]], render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)
timer.start()
```

**설명**:
- **timer.pause()**: Densification 시간은 학습 시간에서 제외
- **training_report()**: TensorBoard에 메트릭 기록
- **주기적 저장**: `saving_iterations`에 지정된 iteration에서 모델 저장
- **렌더링 이미지 저장**: 학습 과정 추적을 위해 주기적으로 렌더링 이미지 저장
  - 초기 (iteration < 1000): 10 iteration마다
  - 중기 (iteration < 3000): 50 iteration마다
  - 후기 (iteration < 60000): 100 iteration마다

---

## 2단계: Densification 수행 (263-290번 줄)

### 2.1 Densification 조건 확인 (264번 줄)

```python
if iteration < opt.densify_until_iter:
```

**의미**:
- `densify_until_iter`까지만 densification 수행
- 이후에는 Gaussian 구조를 고정하고 파라미터만 최적화

### 2.2 Radii 및 통계 업데이트 (265-267번 줄)

```python
# Keep track of max radii in image-space for pruning
gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
```

**설명**:

1. **max_radii2D 업데이트**:
   - 각 Gaussian의 **최대 2D 반경**을 추적
   - 여러 뷰에서 본 최대 크기를 기록
   - Pruning 시 "너무 큰 Gaussian" 판단에 사용

2. **add_densification_stats**:
   - Gradient 통계 수집
   - `xyz_gradient_accum`: 각 Gaussian의 gradient 누적 합
   - `denom`: 각 Gaussian이 보인 횟수
   - 나중에 평균 gradient 계산: `grads = xyz_gradient_accum / denom`

### 2.3 Threshold 계산 (269-274번 줄)

```python
if stage == "coarse":
    opacity_threshold = opt.opacity_threshold_coarse
    densify_threshold = opt.densify_grad_threshold_coarse
else:    
    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter)
```

**설명**:

- **Coarse Stage**: 고정된 threshold 사용
  - 초기 단계이므로 일관된 기준 필요

- **Fine Stage**: **선형 감소 (Linear Decay)**
  - 초기에는 높은 threshold → 적은 densification
  - 점진적으로 낮은 threshold → 더 많은 densification
  - **공식**: `threshold = init - (init - final) * (iteration / densify_until_iter)`
  
  **예시**:
  ```
  opacity_threshold_fine_init = 0.01
  opacity_threshold_fine_after = 0.005
  densify_until_iter = 30000
  
  iteration = 0:    opacity_threshold = 0.01
  iteration = 15000: opacity_threshold = 0.0075
  iteration = 30000: opacity_threshold = 0.005
  ```

**왜 선형 감소를 사용하는가?**
- 초기: 큰 구조부터 학습 → 높은 threshold로 선택적 densification
- 후기: 세부 표현 개선 → 낮은 threshold로 더 많은 densification

### 2.4 Densify 실행 (275-278번 줄)

```python
if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0] < 360000:
    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
    
    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
```

**실행 조건**:

1. **`iteration > densify_from_iter`**: 
   - 초기 몇 iteration은 건너뜀
   - 초기에는 구조가 불안정하므로 densification 지연

2. **`iteration % densification_interval == 0`**:
   - 주기적으로 실행 (예: 100 iteration마다)
   - 매 iteration마다 하면 비용이 큼

3. **`gaussians.get_xyz.shape[0] < 360000`**:
   - 최대 360,000개 제한
   - 메모리 및 계산 비용 제어

**파라미터**:
- `size_threshold = 20`: 
  - `iteration > opacity_reset_interval`일 때만 설정
  - 화면에서 20픽셀 이상인 Gaussian은 pruning 대상
- `densify()` 함수 호출:
  - `densify_threshold`: Gradient threshold
  - `opacity_threshold`: 불투명도 threshold
  - `scene.cameras_extent`: Scene의 크기
  - `size_threshold`: 화면 크기 threshold
  - `5, 5`: density_threshold, displacement_scale (사용되지 않을 수 있음)

**Densify 내부 동작**:
1. 평균 gradient 계산: `grads = xyz_gradient_accum / denom`
2. **Clone**: 작은 Gaussian 복제
3. **Split**: 큰 Gaussian 분할

### 2.5 Prune 실행 (279-282번 줄)

```python
if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0] > 200000:
    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
    
    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
```

**실행 조건**:

1. **`iteration > pruning_from_iter`**: 
   - Densification보다 늦게 시작
   - 충분한 Gaussian이 생성된 후 pruning 시작

2. **`iteration % pruning_interval == 0`**:
   - 주기적으로 실행

3. **`gaussians.get_xyz.shape[0] > 200000`**:
   - 200,000개 이상일 때만 실행
   - 초기에는 Gaussian이 부족하므로 pruning 하지 않음

**Prune 대상**:
- **낮은 불투명도**: `opacity < opacity_threshold`
- **너무 큰 크기**: 
  - 2D: `max_radii2D > size_threshold` (화면에서 20픽셀 이상)
  - 3D: `scaling > 0.1 * cameras_extent`

**의미**:
- Densification으로 Gaussian이 증가 → Pruning으로 불필요한 것 제거
- 밀도와 품질의 균형 유지

### 2.6 Grow 실행 (285-287번 줄)

```python
if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0] < 360000 and opt.add_point:
    gaussians.grow(5,5,scene.model_path,iteration,stage)
```

**실행 조건**:

1. **`iteration % densification_interval == 0`**: Densification과 같은 주기
2. **`gaussians.get_xyz.shape[0] < 360000`**: 최대 개수 제한
3. **`opt.add_point`**: 옵션으로 활성화/비활성화 가능

**의미**:
- `grow()`는 추가적인 Gaussian 생성 방법
- Densification (Clone/Split)과는 별도로 동작
- 구현에 따라 다를 수 있음 (현재 코드베이스에서 정확한 구현 확인 필요)

### 2.7 Opacity Reset (288-290번 줄)

```python
if iteration % opt.opacity_reset_interval == 0:
    print("reset opacity")
    gaussians.reset_opacity()
```

**의미**:
- 주기적으로 opacity를 리셋
- **목적**: 
  - 학습 중 opacity가 잘못 학습된 경우 초기화
  - 새로운 Gaussian이 제대로 보이도록 보장
  - 일반적으로 `opacity_reset_interval = 3000` 정도

**동작**:
- `reset_opacity()` 함수가 opacity 값을 초기값으로 리셋
- 구체적인 구현은 `gaussian_model.py`에 있음

---

## 전체 Densification 프로세스 요약

### 각 Iteration에서의 순서

```
1. 렌더링 (185-197번 줄)
   ↓
2. Loss 계산 및 Backward (206-230번 줄)
   ↓
3. torch.no_grad() 컨텍스트 시작 (233번 줄)
   ↓
4. Progress Bar 업데이트 (235-242번 줄)
   ↓
5. 로깅 및 저장 (247-261번 줄)
   ↓
6. Densification 수행 (263-290번 줄)
   ├─ Radii 및 통계 업데이트 (265-267번 줄)
   ├─ Threshold 계산 (269-274번 줄)
   ├─ Densify 실행 (275-278번 줄) [조건부]
   ├─ Prune 실행 (279-282번 줄) [조건부]
   ├─ Grow 실행 (285-287번 줄) [조건부]
   └─ Opacity Reset (288-290번 줄) [조건부]
   ↓
7. Optimizer Step (295-297번 줄)
```

### Densification 주기 예시

**가정**: 
- `densify_from_iter = 500`
- `densification_interval = 100`
- `pruning_from_iter = 1000`
- `pruning_interval = 100`
- `opacity_reset_interval = 3000`

**실행 시점**:

| Iteration | Densify | Prune | Grow | Opacity Reset |
|-----------|---------|-------|------|---------------|
| 100 | ❌ | ❌ | ❌ | ❌ |
| 500 | ❌ | ❌ | ❌ | ❌ |
| 600 | ✅ | ❌ | ✅ | ❌ |
| 700 | ❌ | ❌ | ❌ | ❌ |
| 1000 | ✅ | ❌ | ✅ | ❌ |
| 1100 | ❌ | ✅ | ❌ | ❌ |
| 3000 | ✅ | ✅ | ✅ | ✅ |

---

## 핵심 파라미터 설명

### Threshold 파라미터

| 파라미터 | 의미 | 사용 위치 |
|---------|------|----------|
| `opacity_threshold` | 불투명도 기준 | Prune, Clone/Split 선택 |
| `densify_threshold` | Gradient 기준 | Clone/Split 대상 선택 |
| `size_threshold` | 화면 크기 기준 (픽셀) | Prune 대상 선택 |

### Iteration 파라미터

| 파라미터 | 의미 | 일반값 |
|---------|------|--------|
| `densify_until_iter` | Densification 종료 시점 | 30000 |
| `densify_from_iter` | Densification 시작 시점 | 500 |
| `densification_interval` | Densification 주기 | 100 |
| `pruning_from_iter` | Pruning 시작 시점 | 1000 |
| `pruning_interval` | Pruning 주기 | 100 |
| `opacity_reset_interval` | Opacity 리셋 주기 | 3000 |

### 개수 제한

| 제한 | 값 | 의미 |
|------|-----|------|
| 최대 Gaussian 개수 | 360,000 | Densify/Grow 실행 조건 |
| 최소 Gaussian 개수 (Pruning) | 200,000 | Prune 실행 조건 |

---

## 주의사항

1. **순서의 중요성**:
   - Densify → Prune → Grow 순서로 실행
   - 각 단계가 다음 단계에 영향을 줌

2. **조건부 실행**:
   - 모든 densification 작업이 조건부로 실행됨
   - 조건을 만족하지 않으면 해당 iteration에서 건너뜀

3. **메모리 관리**:
   - `torch.cuda.empty_cache()`는 주석 처리됨 (287번 줄)
   - 필요시 수동으로 호출 가능

4. **Stage별 차이**:
   - Coarse/Fine stage에 따라 threshold 계산 방식이 다름
   - Fine stage에서는 선형 감소 사용

---

## 참고

- 더 자세한 Clone/Split/Prune 동작은 `DENSIFICATION_PROCESS.md` 참고
- `gaussian_model.py`의 `densify()`, `prune()`, `grow()` 메서드 구현 확인

