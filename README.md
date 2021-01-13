## **PoseFix**
---

#### [개요]
- [[Paper]](https://arxiv.org/abs/1812.03595) | [[Code]](https://github.com/mks0601/PoseFix_RELEASE)
    ![pipeline](./imgs/pipeline.png)
    - Pose Estimator가 아니라  `Refining을 하는  model-agnotic 모델`임.
    - 기존에 존재하던 Pose Refine 모델들은  Two-Stage였음 → Pose estimiator에 의존적 이었고 따라서 Refinment를 성공하기 위해서는 세심한 설계가 필요했음
    - 저자는 Pose Estimator 모델 아키텍처와 상관 없이 성공적인 결과를 내는 모델을 만드는게 목적임
<br></br>
<br></br>

#### [학습 Keypoint]
- OKS(Object Keypoint Similarity), KS(Keypoint similarity), Jitter, Inversion, Swap, Miss와 같은 pose estimiaton 에러를 사용해 Synthesized한 데이터를 만들고 이를 학습에 활용하여 Pose Refinement model을 학습하였음.

    ![Keypoint1](./imgs/Keypoint_2.png)

    ![Keypoint2](./imgs/Keypoint_1.png)

- OKS
    - Object Keypoint Similarity로 COCO Dataset에서 정의한 metric
    - 예측한 keypoint와 GT keypoint와의 유사도를 측정하는 방법

- Keypoint Similarity
- Jitter: estimator 결과가 in-approximat 안에 존재하지만 human error margin 밖에 존재할 때
- Inverseion: estimator 결과가 잘못된 신체 부위에 있는 경우
- Swap: estimator 결과가 다른 사람에게 존재
- Miss: estimator 결과가 in-approximat 안에 존재하지 않을 때
- [[Ref]](https://arxiv.org/abs/1707.05388) 해당 논문에 나오는 분포를 참고해서 Synthesized한 데이터 만들었음
    - frequency of each pose error(Jitter, Inversion, Swap, Miss) according to each pose error
    - the number of visible keypoints
    - overlap in the input image
<br></br>
<br></br>

#### [학습과정]
- 논문 구현을 중 `백본 모델만 ResNet-152가 아닌 ResNet-50으로 구현` 후 학습
- 결과가 생각보다 좋지 않았음
    -  해결하고자 하는 태스크
        1. 사람-사람 occlusion
        2. 사람-배경 occlusion

        사람-사람의 경우는 detection 자체가 안되는 경우가 많아서 `타깃으로 잡은 것은 2번의 사람-배경의 경우`인데  테스트 환경이 주로 매장인 특별한 상황
        때문에 하체가 가려지는 경우가 상대적으로 많았고 해당 경우에 상반신만 포즈가 제대로 찍히는 것이 문제 였음.

        → 논문의 PoseFix는 confidence가 높은 경우에는 좋은 방향으로의 수정이 되었지만 낮은 경우는 여전히 별다른 성과를 보이지 못했음

    - Augmentation의 방법 변화
        - 원 논문은 Flip, Rotation, Synthesize 세 가지를 수행해줌
        - `추가적으로 하체(발목-무릎, 무릎-엉덩이) Keypoin의 인풋 길이를 80%까지 작게 해주고 cutout을 더해` 하체가 occlusion에의해 가려졌을 때 예측되는 경우를 가상 적으로 만들어 주었음 → **120/140Epochs 학습해봤지만 기존의 PoseFix보다 좋은 결과가 나오지는 않는다.**
<br></br>
<br></br>

#### [추론속도]
- FPS → 최저 27 samples/s   최대 30 samples/s
    - 테스트 환경
        - `CPU`: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz
        - `GPU`: GeForce RTX 2070 SUPER
        - `Memory`: 16GB
        - `Image size`: 384x288x3
        - `Batch`: 16

- 해당 추론 속도는 배치 사이즈가 확보 되었을 때 나오는 속도
    - Posefix인풋으로 original모델의 아웃풋 좌표들을 heatmap으로 매핑해줘야하는데 해당 부분에서 병목현상 발생
<br></br>
<br></br>



#### [디렉터리 구조]
![Directory](./imgs/Directory.png)
<br></br>


#### [사용법]
- Train
``` shell
    python3 train.py --train_batch --test_batch --flip_test
```

- Test
``` shell
    python3 test.py --checkpoint --test_batch --flip_test --video_path --detection_json
```
