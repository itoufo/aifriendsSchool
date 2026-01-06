# AIインフラ・アーキテクチャ設計

![AIインフラストラクチャ](/images/illustrations/level4-infrastructure.jpg)

## 学習目標
- エンタープライズグレードのAIシステムアーキテクチャを設計する
- スケーラブルなMLパイプラインを構築・運用する
- クラウドネイティブなAIインフラを最適化する
- 分散学習とエッジコンピューティングを実装する

## 想定学習時間
約10-12時間（実装演習含む）

---

## 1. エンタープライズAIアーキテクチャ

### システム設計の原則

#### 1. マイクロサービスベースのAIアーキテクチャ
```python
# AIマイクロサービスアーキテクチャ設計
class AIServiceArchitecture:
    def __init__(self):
        self.services = {
            "data_ingestion": {
                "purpose": "データ収集・前処理",
                "technologies": ["Apache Kafka", "AWS Kinesis", "Apache NiFi"],
                "scalability": "水平スケーリング",
                "interfaces": ["REST API", "gRPC", "WebSocket"]
            },
            "feature_engineering": {
                "purpose": "特徴量生成・変換",
                "technologies": ["Apache Spark", "Dask", "Ray"],
                "scalability": "分散処理",
                "storage": ["Feature Store", "Redis", "DynamoDB"]
            },
            "model_serving": {
                "purpose": "推論エンドポイント",
                "technologies": ["TensorFlow Serving", "TorchServe", "MLflow"],
                "deployment": ["Kubernetes", "AWS SageMaker", "Azure ML"],
                "patterns": ["A/Bテスト", "カナリアリリース", "影の実行"]
            },
            "monitoring": {
                "purpose": "性能監視・アラート",
                "technologies": ["Prometheus", "Grafana", "DataDog"],
                "metrics": ["レイテンシ", "スループット", "精度", "ドリフト"],
                "alerting": ["PagerDuty", "Slack", "Email"]
            }
        }
    
    def design_reference_architecture(self, requirements):
        """リファレンスアーキテクチャ設計"""
        architecture = {
            "layers": {
                "presentation": self.design_presentation_layer(requirements),
                "api_gateway": self.design_api_gateway(requirements),
                "business_logic": self.design_business_logic(requirements),
                "ml_services": self.design_ml_services(requirements),
                "data_layer": self.design_data_layer(requirements),
                "infrastructure": self.design_infrastructure(requirements)
            },
            "cross_cutting_concerns": {
                "security": self.design_security_layer(),
                "monitoring": self.design_monitoring_layer(),
                "logging": self.design_logging_layer(),
                "governance": self.design_governance_layer()
            },
            "deployment": self.design_deployment_strategy(requirements)
        }
        
        return architecture
    
    def design_ml_services(self, requirements):
        """ML サービス層設計"""
        return {
            "training_pipeline": {
                "orchestration": "Apache Airflow / Kubeflow",
                "compute": "GPU クラスター / TPU",
                "experiment_tracking": "MLflow / Weights & Biases",
                "model_registry": "MLflow Model Registry / AWS Model Registry"
            },
            "inference_service": {
                "online": {
                    "latency": "< 100ms",
                    "throughput": "> 1000 QPS",
                    "availability": "99.9%",
                    "deployment": "Blue-Green / Canary"
                },
                "batch": {
                    "scheduling": "Cron / Event-driven",
                    "processing": "Spark / Batch Transform",
                    "storage": "S3 / HDFS"
                },
                "streaming": {
                    "framework": "Kafka Streams / Flink",
                    "latency": "< 1s",
                    "processing": "Windowing / Aggregation"
                }
            },
            "model_management": {
                "versioning": "Git LFS / DVC",
                "lifecycle": "Development → Staging → Production",
                "rollback": "自動ロールバック機能",
                "monitoring": "精度モニタリング / ドリフト検出"
            }
        }
```

#### 2. イベント駆動アーキテクチャ
```python
class EventDrivenMLArchitecture:
    def __init__(self):
        self.event_bus_config = {
            "broker": "Apache Kafka",
            "topics": {
                "data-ingestion": {"partitions": 10, "replication": 3},
                "feature-computed": {"partitions": 5, "replication": 3},
                "prediction-request": {"partitions": 20, "replication": 3},
                "prediction-result": {"partitions": 20, "replication": 3},
                "model-update": {"partitions": 1, "replication": 3}
            }
        }
    
    def implement_event_handlers(self):
        """イベントハンドラー実装"""
        handlers = {
            "data_ingestion_handler": """
            @event_handler('data-ingestion')
            async def process_new_data(event):
                # データ検証
                validated_data = await validate_schema(event.data)
                
                # 特徴量計算トリガー
                await publish_event('feature-computation', {
                    'data_id': validated_data.id,
                    'timestamp': datetime.now()
                })
                
                # データレイク保存
                await store_to_data_lake(validated_data)
            """,
            
            "prediction_handler": """
            @event_handler('prediction-request')
            async def handle_prediction(event):
                # 特徴量取得
                features = await feature_store.get(event.feature_ids)
                
                # モデル推論
                model = await model_registry.get_latest(event.model_name)
                prediction = await model.predict(features)
                
                # 結果公開
                await publish_event('prediction-result', {
                    'request_id': event.request_id,
                    'prediction': prediction,
                    'confidence': prediction.confidence,
                    'model_version': model.version
                })
                
                # メトリクス記録
                await metrics.record_prediction(prediction)
            """,
            
            "model_update_handler": """
            @event_handler('model-update')
            async def update_model(event):
                # 新モデル検証
                validation_result = await validate_model(event.model)
                
                if validation_result.passed:
                    # カナリアデプロイ
                    await deploy_canary(event.model, traffic_percentage=10)
                    
                    # メトリクス監視
                    await monitor_canary_metrics(event.model.id)
                    
                    # 段階的ロールアウト
                    await schedule_progressive_rollout(event.model.id)
                else:
                    await alert_team(validation_result.errors)
            """
        }
        
        return handlers
```

### 実践的な実装パターン

#### 1. サーキットブレーカーパターン
```python
import asyncio
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60, expected_exception=Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func, *args, **kwargs):
        """サーキットブレーカー経由での関数呼び出し"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self):
        return (self.last_failure_time and 
                datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout))
    
    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# 使用例
class MLServiceClient:
    def __init__(self):
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30,
            expected_exception=TimeoutError
        )
    
    async def predict(self, data):
        """推論APIコール（サーキットブレーカー付き）"""
        return await self.circuit_breaker.call(self._make_prediction, data)
    
    async def _make_prediction(self, data):
        # 実際のAPI呼び出し
        async with aiohttp.ClientSession() as session:
            async with session.post('http://ml-service/predict', json=data) as response:
                if response.status != 200:
                    raise TimeoutError("Service unavailable")
                return await response.json()
```

---

## 2. MLOpsパイプライン構築

### CI/CD for Machine Learning

#### 1. 自動化されたMLパイプライン
```yaml
# ml-pipeline.yaml - Kubeflow Pipeline定義
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: ml-training-pipeline-
spec:
  entrypoint: ml-pipeline
  templates:
  - name: ml-pipeline
    dag:
      tasks:
      - name: data-validation
        template: validate-data
        arguments:
          parameters:
          - name: dataset-path
            value: "s3://ml-data/training-data"
      
      - name: feature-engineering
        dependencies: [data-validation]
        template: compute-features
        arguments:
          parameters:
          - name: validated-data
            value: "{{tasks.data-validation.outputs.parameters.validated-path}}"
      
      - name: model-training
        dependencies: [feature-engineering]
        template: train-model
        arguments:
          parameters:
          - name: features
            value: "{{tasks.feature-engineering.outputs.parameters.feature-path}}"
      
      - name: model-evaluation
        dependencies: [model-training]
        template: evaluate-model
        arguments:
          parameters:
          - name: model-path
            value: "{{tasks.model-training.outputs.parameters.model-path}}"
      
      - name: deploy-decision
        dependencies: [model-evaluation]
        template: deployment-gate
        when: "{{tasks.model-evaluation.outputs.parameters.accuracy}} > 0.95"
      
      - name: model-deployment
        dependencies: [deploy-decision]
        template: deploy-model
        arguments:
          parameters:
          - name: model-path
            value: "{{tasks.model-training.outputs.parameters.model-path}}"
```

#### 2. テスト自動化フレームワーク
```python
import pytest
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
import great_expectations as ge

class MLTestFramework:
    def __init__(self):
        self.test_suites = {
            "data_tests": self.data_quality_tests,
            "feature_tests": self.feature_engineering_tests,
            "model_tests": self.model_performance_tests,
            "integration_tests": self.integration_tests,
            "drift_tests": self.drift_detection_tests
        }
    
    def data_quality_tests(self, df):
        """データ品質テスト"""
        # Great Expectations による検証
        context = ge.DataContext()
        suite = context.create_expectation_suite("data_quality_suite")
        
        validator = context.get_validator(
            batch_request={"path": df},
            expectation_suite_name="data_quality_suite"
        )
        
        # スキーマ検証
        validator.expect_table_columns_to_match_set(
            column_set=["feature1", "feature2", "label"]
        )
        
        # NULL値チェック
        validator.expect_column_values_to_not_be_null("label")
        
        # 値の範囲チェック
        validator.expect_column_values_to_be_between(
            "feature1", min_value=0, max_value=1
        )
        
        # カテゴリ値チェック
        validator.expect_column_values_to_be_in_set(
            "category", ["A", "B", "C"]
        )
        
        return validator.validate()
    
    def model_performance_tests(self, model, test_data):
        """モデル性能テスト"""
        X_test, y_test = test_data
        predictions = model.predict(X_test)
        
        tests = {
            "accuracy_threshold": {
                "metric": accuracy_score(y_test, predictions),
                "threshold": 0.95,
                "passed": None
            },
            "precision_threshold": {
                "metric": precision_score(y_test, predictions, average='weighted'),
                "threshold": 0.93,
                "passed": None
            },
            "recall_threshold": {
                "metric": recall_score(y_test, predictions, average='weighted'),
                "threshold": 0.93,
                "passed": None
            },
            "inference_latency": {
                "metric": self.measure_inference_latency(model, X_test[:100]),
                "threshold": 100,  # ms
                "passed": None
            }
        }
        
        # 閾値チェック
        for test_name, test_config in tests.items():
            test_config["passed"] = test_config["metric"] >= test_config["threshold"]
        
        return tests
    
    def drift_detection_tests(self, reference_data, current_data):
        """データドリフト検出テスト"""
        from alibi_detect.cd import KSDrift, ChiSquareDrift
        
        # Kolmogorov-Smirnov テスト（連続値）
        drift_detector_continuous = KSDrift(
            reference_data[:, :5],  # 連続特徴量
            p_val=0.05
        )
        
        # Chi-Square テスト（カテゴリ値）
        drift_detector_categorical = ChiSquareDrift(
            reference_data[:, 5:],  # カテゴリ特徴量
            p_val=0.05
        )
        
        drift_results = {
            "continuous_features": drift_detector_continuous.predict(current_data[:, :5]),
            "categorical_features": drift_detector_categorical.predict(current_data[:, 5:]),
            "action_required": None
        }
        
        # アクション判定
        if drift_results["continuous_features"]["data"]["is_drift"]:
            drift_results["action_required"] = "Model retraining recommended"
        
        return drift_results
```

### 分散学習システム

#### 1. データ並列分散学習
```python
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

class DistributedTrainingSystem:
    def __init__(self, world_size, rank):
        self.world_size = world_size
        self.rank = rank
        self.setup_distributed()
    
    def setup_distributed(self):
        """分散環境セットアップ"""
        dist.init_process_group(
            backend='nccl',  # NVIDIA GPU用
            init_method='env://',
            world_size=self.world_size,
            rank=self.rank
        )
        
        # GPUデバイス設定
        torch.cuda.set_device(self.rank)
    
    def create_distributed_model(self, model_class, *args, **kwargs):
        """分散モデル作成"""
        model = model_class(*args, **kwargs).cuda(self.rank)
        
        # DDP でラップ
        ddp_model = DDP(
            model, 
            device_ids=[self.rank],
            output_device=self.rank,
            find_unused_parameters=True
        )
        
        return ddp_model
    
    def distributed_training_loop(self, model, dataloader, optimizer, epochs):
        """分散学習ループ"""
        model.train()
        
        for epoch in range(epochs):
            # データローダーのサンプラー設定（各GPUで異なるデータ）
            dataloader.sampler.set_epoch(epoch)
            
            epoch_loss = 0
            for batch_idx, (data, target) in enumerate(dataloader):
                data = data.cuda(self.rank, non_blocking=True)
                target = target.cuda(self.rank, non_blocking=True)
                
                optimizer.zero_grad()
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                
                # 勾配の全プロセス平均化
                self.average_gradients(model)
                
                optimizer.step()
                epoch_loss += loss.item()
            
            # エポック終了時の同期
            dist.barrier()
            
            # ロスの全プロセス平均
            avg_loss = self.reduce_value(epoch_loss / len(dataloader))
            
            if self.rank == 0:
                print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
    
    def average_gradients(self, model):
        """勾配の平均化"""
        for param in model.parameters():
            if param.requires_grad:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size
    
    def reduce_value(self, value):
        """値の全プロセス集約"""
        tensor = torch.tensor(value).cuda(self.rank)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item() / self.world_size

# 使用例
def main():
    # 環境変数から設定取得
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    
    # 分散学習システム初期化
    trainer = DistributedTrainingSystem(world_size, rank)
    
    # モデル作成
    model = trainer.create_distributed_model(ResNet50, num_classes=1000)
    
    # データローダー作成（分散サンプラー使用）
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=256,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # 最適化
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    
    # 学習実行
    trainer.distributed_training_loop(model, train_loader, optimizer, epochs=100)
```

---

## 3. クラウドネイティブAI実装

### Kubernetes上のAIワークロード

#### 1. カスタムリソース定義（CRD）
```yaml
# ml-training-job-crd.yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: mltrainingjobs.ml.company.com
spec:
  group: ml.company.com
  versions:
  - name: v1
    served: true
    storage: true
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              modelType:
                type: string
                enum: ["tensorflow", "pytorch", "xgboost"]
              dataSource:
                type: object
                properties:
                  s3Bucket:
                    type: string
                  path:
                    type: string
              resources:
                type: object
                properties:
                  gpu:
                    type: integer
                    minimum: 0
                    maximum: 8
                  memory:
                    type: string
                  cpu:
                    type: string
              hyperparameters:
                type: object
                additionalProperties:
                  type: string
              outputPath:
                type: string
          status:
            type: object
            properties:
              phase:
                type: string
                enum: ["Pending", "Running", "Succeeded", "Failed"]
              startTime:
                type: string
              completionTime:
                type: string
              message:
                type: string
  scope: Namespaced
  names:
    plural: mltrainingjobs
    singular: mltrainingjob
    kind: MLTrainingJob
```

#### 2. Operatorパターン実装
```python
import kopf
import kubernetes
from kubernetes import client, config

@kopf.on.create('ml.company.com', 'v1', 'mltrainingjobs')
async def create_training_job(spec, name, namespace, **kwargs):
    """MLトレーニングジョブ作成ハンドラー"""
    
    # Kubernetes API クライアント
    config.load_incluster_config()
    v1 = client.CoreV1Api()
    batch_v1 = client.BatchV1Api()
    
    # ジョブ仕様作成
    job = create_k8s_job_spec(name, namespace, spec)
    
    # ConfigMap作成（ハイパーパラメータ）
    configmap = create_hyperparameter_configmap(name, namespace, spec['hyperparameters'])
    v1.create_namespaced_config_map(namespace, configmap)
    
    # PVC作成（データボリューム）
    pvc = create_data_volume_pvc(name, namespace, spec['dataSource'])
    v1.create_namespaced_persistent_volume_claim(namespace, pvc)
    
    # Job作成
    response = batch_v1.create_namespaced_job(namespace, job)
    
    # ステータス更新
    return {'phase': 'Running', 'startTime': datetime.now().isoformat()}

@kopf.on.update('ml.company.com', 'v1', 'mltrainingjobs')
async def update_training_job(spec, status, name, namespace, **kwargs):
    """ジョブステータス監視・更新"""
    batch_v1 = client.BatchV1Api()
    
    try:
        job = batch_v1.read_namespaced_job(name, namespace)
        
        if job.status.succeeded:
            # モデルレジストリに登録
            register_model_to_registry(name, namespace, spec['outputPath'])
            return {'phase': 'Succeeded', 'completionTime': datetime.now().isoformat()}
        
        elif job.status.failed:
            return {'phase': 'Failed', 'message': 'Training failed'}
    
    except Exception as e:
        return {'phase': 'Failed', 'message': str(e)}

def create_k8s_job_spec(name, namespace, spec):
    """Kubernetes Job仕様生成"""
    return client.V1Job(
        metadata=client.V1ObjectMeta(name=name),
        spec=client.V1JobSpec(
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels={"app": name}),
                spec=client.V1PodSpec(
                    containers=[
                        client.V1Container(
                            name="training",
                            image=f"ml-training:{spec['modelType']}",
                            resources=client.V1ResourceRequirements(
                                requests={
                                    "memory": spec['resources'].get('memory', '4Gi'),
                                    "cpu": spec['resources'].get('cpu', '2'),
                                    "nvidia.com/gpu": spec['resources'].get('gpu', 0)
                                },
                                limits={
                                    "memory": spec['resources'].get('memory', '4Gi'),
                                    "cpu": spec['resources'].get('cpu', '2'),
                                    "nvidia.com/gpu": spec['resources'].get('gpu', 0)
                                }
                            ),
                            volume_mounts=[
                                client.V1VolumeMount(
                                    name="data-volume",
                                    mount_path="/data"
                                ),
                                client.V1VolumeMount(
                                    name="config-volume",
                                    mount_path="/config"
                                )
                            ],
                            env=[
                                client.V1EnvVar(
                                    name="MODEL_OUTPUT_PATH",
                                    value=spec['outputPath']
                                )
                            ]
                        )
                    ],
                    volumes=[
                        client.V1Volume(
                            name="data-volume",
                            persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                                claim_name=f"{name}-data-pvc"
                            )
                        ),
                        client.V1Volume(
                            name="config-volume",
                            config_map=client.V1ConfigMapVolumeSource(
                                name=f"{name}-config"
                            )
                        )
                    ],
                    restart_policy="Never"
                )
            )
        )
    )
```

### サーバーレスML

#### 1. AWS Lambda for ML推論
```python
import json
import boto3
import joblib
import numpy as np

# モデルのグローバルロード（コールドスタート最適化）
s3 = boto3.client('s3')
MODEL = None

def load_model():
    global MODEL
    if MODEL is None:
        # S3からモデルダウンロード
        s3.download_file('ml-models-bucket', 'models/latest.pkl', '/tmp/model.pkl')
        MODEL = joblib.load('/tmp/model.pkl')
    return MODEL

def lambda_handler(event, context):
    """Lambda関数エントリポイント"""
    try:
        # モデルロード
        model = load_model()
        
        # 入力データ解析
        body = json.loads(event['body'])
        features = np.array(body['features']).reshape(1, -1)
        
        # 推論実行
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].tolist()
        
        # レスポンス作成
        response = {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'prediction': int(prediction),
                'probability': probability,
                'model_version': 'v1.2.3'
            })
        }
        
        # メトリクス送信
        send_metrics(prediction, probability)
        
        return response
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def send_metrics(prediction, probability):
    """CloudWatchメトリクス送信"""
    cloudwatch = boto3.client('cloudwatch')
    
    cloudwatch.put_metric_data(
        Namespace='ML/Inference',
        MetricData=[
            {
                'MetricName': 'PredictionConfidence',
                'Value': max(probability),
                'Unit': 'None',
                'Timestamp': datetime.now()
            },
            {
                'MetricName': 'InferenceCount',
                'Value': 1,
                'Unit': 'Count',
                'Timestamp': datetime.now()
            }
        ]
    )
```

---

## 4. エッジAIとフェデレーテッド学習

### エッジデプロイメント

#### 1. TensorFlow Lite変換と最適化
```python
import tensorflow as tf
import tensorflow_model_optimization as tfmot

class EdgeModelOptimizer:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
    
    def quantize_model(self):
        """量子化によるモデル圧縮"""
        # Quantization Aware Training
        quantize_model = tfmot.quantization.keras.quantize_model
        
        q_aware_model = quantize_model(self.model)
        q_aware_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return q_aware_model
    
    def prune_model(self, target_sparsity=0.5):
        """プルーニングによるモデル圧縮"""
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=target_sparsity,
                begin_step=0,
                end_step=1000
            )
        }
        
        pruned_model = prune_low_magnitude(self.model, **pruning_params)
        
        return pruned_model
    
    def convert_to_tflite(self, quantize=True):
        """TensorFlow Lite形式への変換"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantize:
            # INT8量子化
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            
            # 代表データセット設定（キャリブレーション用）
            def representative_dataset():
                for _ in range(100):
                    data = np.random.rand(1, 224, 224, 3).astype(np.float32)
                    yield [data]
            
            converter.representative_dataset = representative_dataset
        
        # 変換実行
        tflite_model = converter.convert()
        
        # モデル保存
        with open('model.tflite', 'wb') as f:
            f.write(tflite_model)
        
        return tflite_model
    
    def benchmark_edge_model(self, tflite_model_path):
        """エッジモデルのベンチマーク"""
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # 推論時間測定
        import time
        times = []
        
        for _ in range(100):
            input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
            
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        return {
            'avg_inference_time_ms': np.mean(times) * 1000,
            'model_size_mb': os.path.getsize(tflite_model_path) / (1024 * 1024),
            'p95_latency_ms': np.percentile(times, 95) * 1000
        }
```

### フェデレーテッド学習実装

#### 1. FL サーバー・クライアント実装
```python
import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class FederatedLearningServer:
    def __init__(self, model, strategy="FedAvg"):
        self.model = model
        self.strategy = self.create_strategy(strategy)
    
    def create_strategy(self, strategy_name):
        """FL戦略作成"""
        if strategy_name == "FedAvg":
            return fl.server.strategy.FedAvg(
                fraction_fit=0.3,  # 各ラウンドで参加するクライアントの割合
                fraction_evaluate=0.2,
                min_fit_clients=10,  # 最小参加クライアント数
                min_evaluate_clients=5,
                min_available_clients=10,
                evaluate_fn=self.get_evaluate_fn(),
                on_fit_config_fn=self.fit_config,
                on_evaluate_config_fn=self.evaluate_config,
                initial_parameters=fl.common.ndarrays_to_parameters(
                    [val.cpu().numpy() for val in self.model.state_dict().values()]
                )
            )
        elif strategy_name == "FedProx":
            return fl.server.strategy.FedProx(
                proximal_mu=0.1,  # Proximal項の重み
                fraction_fit=0.3,
                fraction_evaluate=0.2,
                min_fit_clients=10,
                min_evaluate_clients=5,
                min_available_clients=10
            )
        
    def get_evaluate_fn(self):
        """サーバー側評価関数"""
        def evaluate(server_round, parameters, config):
            # グローバルモデル評価
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.model.load_state_dict(state_dict, strict=True)
            
            # テストデータで評価
            loss, accuracy = self.test_global_model()
            
            return loss, {"accuracy": accuracy}
        
        return evaluate
    
    def start_server(self, num_rounds=10):
        """FLサーバー起動"""
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=self.strategy
        )

class FederatedLearningClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
    
    def get_parameters(self, config):
        """モデルパラメータ取得"""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters):
        """モデルパラメータ設定"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """ローカル学習"""
        self.set_parameters(parameters)
        
        # 差分プライバシー適用
        privacy_engine = self.setup_differential_privacy()
        
        # ローカル学習実行
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.model.train()
        
        for epoch in range(config["local_epochs"]):
            for batch in self.train_loader:
                data, target = batch
                data = data.to(self.device)
                target = target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                
                # 勾配クリッピング（差分プライバシー）
                if privacy_engine:
                    privacy_engine.clip_gradients()
                
                optimizer.step()
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        """ローカル評価"""
        self.set_parameters(parameters)
        
        loss, accuracy = self.test_local_model()
        
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}
    
    def setup_differential_privacy(self, epsilon=1.0, delta=1e-5):
        """差分プライバシー設定"""
        from opacus import PrivacyEngine
        
        privacy_engine = PrivacyEngine()
        
        model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            epochs=1,
            target_epsilon=epsilon,
            target_delta=delta,
            max_grad_norm=1.0
        )
        
        return privacy_engine

# クライアント起動
def start_client():
    model = create_model()
    train_loader, test_loader = load_local_data()
    
    client = FederatedLearningClient(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client
    )
```

---

## まとめ

### 本章で習得したスキル

1. **アーキテクチャ設計**: エンタープライズグレードのAIシステム設計
2. **MLOps実装**: 自動化されたMLパイプラインとCI/CD
3. **クラウドネイティブ**: Kubernetes上でのAIワークロード管理
4. **エッジ&FL**: エッジデプロイメントとフェデレーテッド学習

### 実装のベストプラクティス

#### インフラストラクチャ
- **Infrastructure as Code**: Terraform/Pulumi使用
- **モニタリング**: Prometheus + Grafana
- **ロギング**: ELK Stack / Fluentd
- **トレーシング**: Jaeger / Zipkin

#### セキュリティ
- **ゼロトラスト**: すべての通信を検証
- **暗号化**: 転送時・保存時の暗号化
- **アクセス制御**: RBAC / ABAC
- **監査ログ**: 全操作の記録

#### パフォーマンス
- **キャッシング**: Redis / Memcached
- **非同期処理**: Message Queue活用
- **自動スケーリング**: HPA / VPA
- **負荷分散**: Ingress / Service Mesh

---

## 参考資料
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [MLOps: Continuous delivery and automation pipelines in ML](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Distributed Training with PyTorch](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)