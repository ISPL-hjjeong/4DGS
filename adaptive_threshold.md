# Adaptive Threshold 및 Gaussian 개수 제어 가이드

## 개요

기존의 고정된 threshold 대신, **학습 가능한 adaptive threshold**와 **Gaussian 생성 개수 제어** 기능을 추가했습니다.

## 주요 기능

### 1. Adaptive Threshold 학습

고정된 threshold 대신, Scene 상태를 기반으로 최적의 threshold를 예측합니다.

#### 방법 1: Per-Gaussian Threshold
- 각 Gaussian마다 다른 threshold를 예측
- Gaussian의 gradient, scaling, opacity 등을 고려하여 개별적으로 threshold 결정
- 더 세밀한 제어 가능

#### 방법 2: Global Threshold
- Scene 전체에 대해 하나의 threshold를 예측
- iteration, 현재 Gaussian 개수, 평균 gradient 등을 고려
- 계산 비용이 낮음

### 2. Gaussian 개수 제어

생성되는 Gaussian의 개수를 제어하는 두 가지 방법:

#### 방법 1: Budget 제어
- 최대 생성 개수를 직접 지정
- 예: `max_densify_budget=1000` → 최대 1000개만 생성
- 메모리 및 계산 비용 제어에 유용

#### 방법 2: Importance 기반 선택
- Gradient magnitude를 importance score로 사용
- 상위 N%만 선택 (기본 50%)
- 가장 중요한 Gaussian만 생성

## 사용 방법

### 1. Arguments 설정

`arguments/__init__.py`의 `ModelHiddenParams` 클래스에서 설정:

```python
# Adaptive Threshold 옵션
self.use_adaptive_threshold = True  # Adaptive threshold 사용
self.adaptive_threshold_mode = 'per_gaussian'  # 'per_gaussian' or 'global'
self.adaptive_threshold_hidden_dim = 64  # Threshold network hidden dimension
self.adaptive_threshold_num_layers = 2  # Threshold network layer 수

# Gaussian 개수 제어 옵션
self.max_densify_budget = 1000  # 최대 생성 개수 제한 (None이면 제한 없음)
self.use_importance_selection = True  # Importance 기반 선택 사용
self.importance_selection_ratio = 0.5  # 상위 50%만 선택
```

`OptimizationParams` 클래스에서 학습률 설정:

```python
self.adaptive_threshold_lr = 0.0001  # Adaptive threshold 네트워크 학습률
```

### 2. Command Line 사용

```bash
python train.py \
    --source_path <path> \
    --model_path <path> \
    --use_adaptive_threshold \
    --adaptive_threshold_mode per_gaussian \
    --max_densify_budget 1000 \
    --use_importance_selection \
    --importance_selection_ratio 0.5
```

## 구현 세부사항

### Adaptive Threshold Network

**입력 (Per-Gaussian 모드):**
- Gradient magnitude: [N]
- Scaling (x, y, z): [N, 3]
- Opacity: [N]
- Iteration (정규화): [N]
- 현재 Gaussian 개수 (정규화): [N]
- Scene extent (정규화): [N]
- **총 8차원**

**입력 (Global 모드):**
- Iteration (정규화): [1]
- 현재 Gaussian 개수 (정규화): [1]
- 평균 gradient (정규화): [1]
- Scene extent (정규화): [1]
- Stage (coarse=0, fine=1): [1]
- **총 5차원**

**출력:**
- Threshold 값: [N, 1] or [1, 1]
- Sigmoid 활성화로 0~1 정규화 후, `base_threshold_range`로 스케일링
- 기본 범위: (0.0002, 0.01)

### Importance 기반 선택

1. **Gradient magnitude 계산**: 각 Gaussian의 gradient 크기
2. **정렬**: Gradient magnitude 기준으로 내림차순 정렬
3. **Top-K 선택**:
   - Budget 제어: 상위 `max_densify_budget`개 선택
   - Importance 선택: 상위 `importance_selection_ratio * N`개 선택

### Budget 제어

- `max_densify_budget`가 설정되면, Importance 기반으로 상위 K개만 선택
- Clone과 Split 각각에 독립적으로 적용
- 예: `max_densify_budget=1000`이면 Clone에서 최대 1000개, Split에서 최대 1000개 생성

## 장점

### 1. Adaptive Threshold의 장점

- **학습 가능**: Scene 상태에 따라 최적의 threshold 자동 조정
- **유연성**: Per-Gaussian 모드로 각 Gaussian에 맞는 threshold 제공
- **효율성**: Global 모드로 계산 비용 절감

### 2. 개수 제어의 장점

- **메모리 제어**: Budget 제어로 메모리 사용량 예측 가능
- **품질 향상**: Importance 기반 선택으로 중요한 Gaussian만 생성
- **학습 안정성**: 과도한 Gaussian 생성 방지

## 실험 추천

### 1. Threshold 학습 검증

```python
# 실험 1: Per-Gaussian vs Global
use_adaptive_threshold=True, adaptive_threshold_mode='per_gaussian'
use_adaptive_threshold=True, adaptive_threshold_mode='global'
use_adaptive_threshold=False  # 기존 방식 (baseline)

# 실험 2: Threshold 범위 조정
base_threshold_range=(0.0001, 0.005)  # 더 낮은 범위
base_threshold_range=(0.0005, 0.02)   # 더 높은 범위
```

### 2. 개수 제어 검증

```python
# 실험 1: Budget 제어 효과
max_densify_budget=None  # 제한 없음 (baseline)
max_densify_budget=500   # 적은 개수
max_densify_budget=2000  # 많은 개수

# 실험 2: Importance 선택 효과
use_importance_selection=False  # 모든 Gaussian (baseline)
use_importance_selection=True, importance_selection_ratio=0.3  # 상위 30%
use_importance_selection=True, importance_selection_ratio=0.7  # 상위 70%
```

### 3. 조합 실험

```python
# 최적 조합 찾기
use_adaptive_threshold=True
adaptive_threshold_mode='per_gaussian'
max_densify_budget=1000
use_importance_selection=True
importance_selection_ratio=0.5
```

## 주의사항

1. **학습률**: Adaptive threshold network의 학습률은 낮게 설정 (기본 0.0001)
2. **초기화**: Threshold network는 Sigmoid로 0~1 출력 후 스케일링하므로, 초기값이 중요
3. **Budget 제한**: 너무 작은 budget은 표현력 저하를 일으킬 수 있음
4. **Importance 선택**: 너무 낮은 ratio는 중요한 Gaussian을 놓칠 수 있음

## 코드 위치

- **AdaptiveThresholdNetwork 클래스**: `scene/gaussian_model.py` (29-175줄)
- **densify_and_clone 메서드**: `scene/gaussian_model.py` (636-720줄)
- **densify_and_split 메서드**: `scene/gaussian_model.py` (722-810줄)
- **Arguments 설정**: `arguments/__init__.py` (106-115줄, 171줄)

## 향후 개선 방향

1. **Temporal Consistency**: 이전/다음 프레임 정보를 threshold 예측에 활용
2. **Loss 기반 학습**: Rendering loss를 기반으로 threshold network 학습
3. **Multi-scale Threshold**: 다른 scale에서 다른 threshold 사용
4. **Dynamic Budget**: 학습 진행에 따라 budget 자동 조정

