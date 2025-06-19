## Language

- [English](#english)
- [中文](#中文)

---

### English

// Your English content here

---

### 中文

## 数据集

我们使用了最常见的五个行为识别数据集，UCF101、Kinetics、HMDB51、SSv2-Small/Full。

## 文件结构

```
Task-Adapter-pp
├─README.md
├─config.yaml 模型参数配置文件
├─dataset.py 数据集
├─module_adapter.py task adapter模型所在
├─module_sem_adapter order adapter模型所在
├─requirements.txt 依赖
├─run.py 入口
├─utils.py 略
├─corpus/
	├─classes_hmdb51.yml
	├─classes_kinetics.yml
	├─classes_somethingcmn.yml
	├─classes_somethingotam.yml
	├─classes_ucf101.yml
```

## 没了