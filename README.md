# chorus_detection

calculate_sims.py: similarity 8가지 방법으로 계산

lyric_structure.py: 가사를 쪼개거나 SSM을 생성하는 함수 포함

ssm_drawing.py: SSM 시각화 함수 포함

실행할 파일 - data_setting.py: 데이터를 가져와서 가사 부분만 자른 뒤 border를 설정하고, SSM을 계산하여 HDF 파일 형태로 저장. / SSM 하나를 시각화 하는 것도 포함


cnn module

__init__.py : 비어있음

nn.py: 공통적인 nn 형식(weight variable, bias variable) 설정 - pytorch에서는 내장 모듈로 해결

dense.py : DNN, mnist_like.py : mnist => 제외

no_padding_1conv.py : 순수한 CNN

joint_mn.py : CNN으로 시작해서 LSTM으로 마무리

=> no_padding_1conv.py만 일단 구현해보고, joint_mn.py는 나중에 필요하면
