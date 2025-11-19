# MLP 기반 Densification 가이드

## 개요

기존의 단순 clone/split 방식 대신, MLP(Neural Network)를 사용하여 더 유동적이고 학습 가능한 방식으로 새로운 Gaussian을 생성하는 기능입니다.

## 주요 차이점

### 기존 방식
- **Clone**: 원본 Gaussian의 속성을 그대로 복사
- **Split**: 정규분포로 샘플링하여 위치 생성, 크기를 고정 비율로 축소

### MLP 기반 방식
- **Clone**: MLP가 원본 속성을 입력받아 새로운 속성 생성 (작은 변위)
- **Split**: MLP가 원본 속성을 입력받아 완전히 새로운 속성 생성 (더 큰 변위)

## 장점

1. **유동적인 생성**: MLP가 학습을 통해 최적의 Gaussian 속성을 생성
2. **컨텍스트 인식**: Gradient, 크기, 위치 등 다양한 정보를 종합적으로 고려
3. **학습 가능**: MLP가 학습되면서 더 나은 densification 전략 학습

## 사용 방법

### 1. 기본 사용

```bash
python train.py \
    --use_mlp_densification \
    --source_path <data_path> \
    --model_path <output_path>
```

### 2. 하이퍼파라미터 조정

```bash
python train.py \
    --use_mlp_densification \
    --mlp_densification_hidden_dim 256 \
    --mlp_densification_num_layers 4 \
    --mlp_densification_lr 0.0002 \
    --source_path <data_path> \
    --model_path <output_path>
```

### 3. Arguments 옵션

#### ModelHiddenParams
- `--use_mlp_densification`: MLP 기반 densification 활성화 (기본: False)
- `--mlp_densification_hidden_dim`: MLP hidden layer 차원 (기본: 128)
- `--mlp_densification_num_layers`: MLP 레이어 수 (기본: 3)

#### OptimizationParams
- `--mlp_densification_lr`: MLP densification 네트워크 학습률 (기본: 0.0001)

## 구현 세부사항

### MLP 네트워크 구조

```
Input Features (32차원)
├─ xyz (3) - Normalized position
├─ scaling (3) - Normalized scaling
├─ rotation (4) - Quaternion
├─ opacity (1)
├─ features (16) - SH features
├─ grad_magnitude (1)
└─ size_flag (1)

↓ MLP (hidden_dim, num_layers)

Output Heads
├─ xyz_head → [N, 3]
├─ scaling_head → [N, 3]
├─ rotation_head → [N, 4]
├─ opacity_head → [N, 1]
└─ features_head → [N, 16]
```

### Clone vs Split

#### Clone (작은 Gaussian)
- 원본 위치에 **작은 변위** 추가
- 원본 속성에 **작은 변화** 추가 (0.1 배율)
- 목적: 빈 영역을 채우기

#### Split (큰 Gaussian)
- 원본 위치 주변에 **더 큰 변위**로 분산
- 원본 크기의 **0.8/N 배**로 축소
- 목적: 세밀한 표현을 위한 분할

## 현재 구현의 특징

### 1. Feature 준비 (`_prepare_mlp_features`)
- Gaussian의 모든 속성을 32차원 feature 벡터로 변환
- Scene extent로 정규화하여 스케일 불변성 확보
- Gradient 정보 포함

### 2. MLP 기반 Clone (`densify_and_clone_mlp`)
- 원본 속성에 MLP 출력의 0.1배를 더하여 새로운 속성 생성
- 작은 변위로 빈 영역 채우기

### 3. MLP 기반 Split (`densify_and_split_mlp`)
- MLP 출력을 직접 사용하여 완전히 새로운 속성 생성
- 원본 위치 주변에 분산 배치

## 주의사항

### 1. MLP 학습
현재 구현에서는 MLP가 `torch.no_grad()` 컨텍스트에서 실행됩니다. 이는:
- **장점**: Densification 속도가 빠름
- **단점**: MLP가 학습되지 않음

**개선 방안**:
- MLP를 학습시키려면 gradient를 전파해야 합니다
- 하지만 densification은 구조 변경 작업이므로 주의가 필요합니다
- 대안: 별도의 loss를 통해 MLP를 학습시키거나, 생성된 Gaussian의 품질을 평가하여 MLP를 학습

### 2. 메모리 사용량
- MLP 네트워크가 추가 메모리를 사용합니다
- Hidden dimension과 layer 수에 따라 메모리 사용량이 증가합니다

### 3. 학습 속도
- MLP forward pass가 추가되어 약간의 오버헤드가 있습니다
- 하지만 일반적으로 densification은 주기적으로만 실행되므로 큰 영향은 없습니다

## 향후 개선 방안

### 1. MLP 학습
```python
# 현재: no_grad() 사용
with torch.no_grad():
    mlp_output = self.mlp_densification_net(mlp_features)

# 개선: gradient 전파
mlp_output = self.mlp_densification_net(mlp_features)
# 생성된 Gaussian의 품질을 평가하는 loss 추가
```

### 2. Attention 메커니즘
- 주변 Gaussian의 정보를 활용하여 더 나은 생성

### 3. 조건부 생성
- Scene의 특정 영역에 따라 다른 전략 사용

### 4. 학습 기반 선택
- Clone vs Split를 MLP가 자동으로 결정

## 예제

### 기본 사용
```python
# train.py 실행 시
python train.py \
    --use_mlp_densification \
    --source_path data/dynerf/bouncingballs \
    --model_path output/mlp_densification_test
```

### 고급 설정
```python
# 더 큰 네트워크 사용
python train.py \
    --use_mlp_densification \
    --mlp_densification_hidden_dim 256 \
    --mlp_densification_num_layers 5 \
    --mlp_densification_lr 0.0002 \
    --source_path data/dynerf/bouncingballs \
    --model_path output/mlp_densification_advanced
```

## 성능 비교

| 방식 | 장점 | 단점 |
|------|------|------|
| **기존 (Clone/Split)** | 빠름, 단순함 | 고정된 규칙, 유연성 부족 |
| **MLP 기반** | 유연함, 학습 가능 | 약간 느림, 메모리 사용 증가 |

## 참고

- `scene/gaussian_model.py`: MLP 네트워크 및 메서드 구현
- `arguments/__init__.py`: 옵션 정의
- `TRAIN_DENSIFICATION_DETAILED.md`: Densification 프로세스 상세 설명

