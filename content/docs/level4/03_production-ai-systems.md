# プロダクション AI システム

![プロダクションAIシステム](/images/illustrations/level4-production-systems.jpg)

## 学習目標
- 本番環境でのAIシステム運用のベストプラクティスを習得する
- モデルのモニタリングとドリフト検出を実装する
- A/Bテストとカナリアデプロイメントを設計する
- 障害対応とインシデント管理体制を構築する

## 想定学習時間
約10-12時間（実装演習・ケーススタディ含む）

---

## 1. プロダクション環境の設計と運用

### MLシステムの信頼性工学

#### 1. SLI/SLO/SLAの定義と実装
```python
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import prometheus_client as prom

@dataclass
class SLIDefinition:
    """Service Level Indicator定義"""
    name: str
    metric_query: str
    aggregation_window: timedelta
    threshold: float
    
@dataclass
class SLOTarget:
    """Service Level Objective定義"""
    sli: SLIDefinition
    target_percentage: float  # e.g., 99.9%
    evaluation_window: timedelta  # e.g., 30 days
    
class MLSystemSLO:
    def __init__(self):
        self.slos = {
            "inference_latency": SLOTarget(
                sli=SLIDefinition(
                    name="p99_latency",
                    metric_query="histogram_quantile(0.99, inference_latency_seconds)",
                    aggregation_window=timedelta(minutes=5),
                    threshold=0.5  # 500ms
                ),
                target_percentage=99.0,
                evaluation_window=timedelta(days=30)
            ),
            "availability": SLOTarget(
                sli=SLIDefinition(
                    name="uptime",
                    metric_query="avg_over_time(up{job='ml_service'}[5m])",
                    aggregation_window=timedelta(minutes=5),
                    threshold=1.0
                ),
                target_percentage=99.9,
                evaluation_window=timedelta(days=30)
            ),
            "prediction_accuracy": SLOTarget(
                sli=SLIDefinition(
                    name="accuracy",
                    metric_query="avg(prediction_accuracy)",
                    aggregation_window=timedelta(hours=1),
                    threshold=0.95
                ),
                target_percentage=99.0,
                evaluation_window=timedelta(days=7)
            )
        }
        
        # Prometheusメトリクス初期化
        self.latency_histogram = prom.Histogram(
            'inference_latency_seconds',
            'Inference latency in seconds',
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
        )
        
        self.accuracy_gauge = prom.Gauge(
            'prediction_accuracy',
            'Model prediction accuracy'
        )
        
        self.error_budget_gauge = prom.Gauge(
            'error_budget_remaining',
            'Remaining error budget percentage',
            ['slo_name']
        )
        
    def calculate_error_budget(self, slo_name: str) -> float:
        """エラーバジェット計算"""
        slo = self.slos[slo_name]
        
        # 実際のSLI達成率を計算（簡略化）
        actual_achievement = self.get_sli_achievement(slo_name)
        
        # エラーバジェット = 100% - SLO目標 + 実際の達成率 - 100%
        error_budget_total = 100 - slo.target_percentage
        error_budget_consumed = slo.target_percentage - actual_achievement
        error_budget_remaining = error_budget_total - error_budget_consumed
        
        # メトリクス更新
        self.error_budget_gauge.labels(slo_name=slo_name).set(error_budget_remaining)
        
        return error_budget_remaining
    
    def should_freeze_deployments(self) -> bool:
        """エラーバジェットに基づくデプロイメント凍結判定"""
        for slo_name in self.slos:
            budget = self.calculate_error_budget(slo_name)
            if budget <= 0:
                return True
        return False

class ReliabilityMonitor:
    """信頼性監視システム"""
    def __init__(self, alert_manager_url: str):
        self.alert_manager_url = alert_manager_url
        self.incidents = []
        
    def check_golden_signals(self) -> Dict[str, bool]:
        """Google SREのGolden Signals監視"""
        signals = {
            "latency": self.check_latency(),
            "traffic": self.check_traffic(),
            "errors": self.check_errors(),
            "saturation": self.check_saturation()
        }
        return signals
    
    def check_latency(self) -> bool:
        """レイテンシ監視"""
        query = """
        histogram_quantile(0.99, 
            rate(http_request_duration_seconds_bucket[5m])
        ) > 0.5
        """
        return self.execute_prometheus_query(query)
    
    def check_traffic(self) -> bool:
        """トラフィック監視"""
        query = """
        rate(http_requests_total[5m]) < 10 or 
        rate(http_requests_total[5m]) > 10000
        """
        return self.execute_prometheus_query(query)
    
    def check_errors(self) -> bool:
        """エラー率監視"""
        query = """
        rate(http_requests_total{status=~"5.."}[5m]) / 
        rate(http_requests_total[5m]) > 0.01
        """
        return self.execute_prometheus_query(query)
    
    def check_saturation(self) -> bool:
        """リソース飽和度監視"""
        query = """
        (
            avg(rate(container_cpu_usage_seconds_total[5m])) > 0.8 or
            avg(container_memory_usage_bytes / container_spec_memory_limit_bytes) > 0.9
        )
        """
        return self.execute_prometheus_query(query)
    
    def create_incident(self, severity: str, description: str):
        """インシデント作成"""
        incident = {
            "id": f"INC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "severity": severity,  # P1, P2, P3, P4
            "description": description,
            "created_at": datetime.now(),
            "status": "open",
            "assignee": self.get_on_call_engineer()
        }
        
        self.incidents.append(incident)
        self.send_alert(incident)
        
        return incident
```

#### 2. サーキットブレーカーとフォールバック
```python
import asyncio
from enum import Enum
from typing import Callable, Any, Optional
import time
import logging

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
        fallback_function: Optional[Callable] = None
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.fallback_function = fallback_function
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.success_count = 0
        self.half_open_threshold = 3
        
        self.logger = logging.getLogger(__name__)
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """サーキットブレーカー経由での関数呼び出し"""
        # OPEN状態チェック
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                self.logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                if self.fallback_function:
                    self.logger.warning("Circuit breaker is OPEN, using fallback")
                    return await self.fallback_function(*args, **kwargs)
                raise Exception("Circuit breaker is OPEN")
        
        # 関数実行
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            
            if self.fallback_function and self.state == CircuitState.OPEN:
                self.logger.warning(f"Primary function failed: {e}, using fallback")
                return await self.fallback_function(*args, **kwargs)
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """リセット試行判定"""
        if self.last_failure_time is None:
            return False
        
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """成功時の処理"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.logger.info("Circuit breaker is now CLOSED")
        else:
            self.failure_count = 0
    
    def _on_failure(self):
        """失敗時の処理"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.logger.error(f"Circuit breaker is now OPEN after {self.failure_count} failures")
    
    def get_state(self) -> Dict:
        """現在の状態取得"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count if self.state == CircuitState.HALF_OPEN else 0
        }

class ModelServiceWithFallback:
    """フォールバック機能付きモデルサービス"""
    def __init__(self, primary_model, fallback_model=None):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30,
            fallback_function=self.fallback_predict if fallback_model else None
        )
        
    async def predict(self, input_data):
        """予測実行（サーキットブレーカー付き）"""
        return await self.circuit_breaker.call(
            self._primary_predict,
            input_data
        )
    
    async def _primary_predict(self, input_data):
        """プライマリモデルでの予測"""
        # タイムアウト設定
        try:
            result = await asyncio.wait_for(
                self.primary_model.predict(input_data),
                timeout=1.0
            )
            return result
        except asyncio.TimeoutError:
            raise Exception("Primary model timeout")
    
    async def fallback_predict(self, input_data):
        """フォールバックモデルでの予測"""
        if self.fallback_model:
            # より軽量・高速なモデルで予測
            return await self.fallback_model.predict(input_data)
        else:
            # デフォルト値を返す
            return {"prediction": "default", "confidence": 0.0}
```

---

## 2. モデルモニタリングとドリフト検出

### 包括的なモニタリングシステム

#### 1. データドリフト検出
```python
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import pandas as pd
from dataclasses import dataclass

@dataclass
class DriftDetectionResult:
    is_drift_detected: bool
    drift_score: float
    feature_drift_scores: Dict[str, float]
    drift_type: str  # "covariate", "prior", "concept"
    recommended_action: str

class DataDriftMonitor:
    def __init__(self, reference_data: pd.DataFrame, 
                 sensitivity: float = 0.05):
        self.reference_data = reference_data
        self.sensitivity = sensitivity
        self.feature_statistics = self._calculate_statistics(reference_data)
        
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict:
        """統計量計算"""
        stats = {}
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                stats[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'quantiles': data[col].quantile([0.25, 0.5, 0.75]).to_dict()
                }
            else:
                stats[col] = {
                    'unique_values': data[col].unique().tolist(),
                    'value_counts': data[col].value_counts().to_dict()
                }
        return stats
    
    def detect_drift(self, current_data: pd.DataFrame) -> DriftDetectionResult:
        """ドリフト検出"""
        feature_drift_scores = {}
        drift_detected = False
        
        for col in current_data.columns:
            if col not in self.reference_data.columns:
                continue
            
            if current_data[col].dtype in ['float64', 'int64']:
                # Kolmogorov-Smirnov test for continuous features
                statistic, p_value = stats.ks_2samp(
                    self.reference_data[col].dropna(),
                    current_data[col].dropna()
                )
                feature_drift_scores[col] = 1 - p_value
                
                if p_value < self.sensitivity:
                    drift_detected = True
            else:
                # Chi-square test for categorical features
                ref_counts = self.reference_data[col].value_counts()
                curr_counts = current_data[col].value_counts()
                
                # Align categories
                all_categories = set(ref_counts.index) | set(curr_counts.index)
                ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]
                
                if sum(curr_aligned) > 0:
                    statistic, p_value = stats.chisquare(
                        curr_aligned,
                        f_exp=[x * sum(curr_aligned) / sum(ref_aligned) 
                               for x in ref_aligned]
                    )
                    feature_drift_scores[col] = 1 - p_value
                    
                    if p_value < self.sensitivity:
                        drift_detected = True
        
        # 全体のドリフトスコア計算
        overall_drift_score = np.mean(list(feature_drift_scores.values()))
        
        # ドリフトタイプの判定
        drift_type = self._determine_drift_type(feature_drift_scores)
        
        # 推奨アクション
        recommended_action = self._recommend_action(
            drift_detected, 
            overall_drift_score, 
            drift_type
        )
        
        return DriftDetectionResult(
            is_drift_detected=drift_detected,
            drift_score=overall_drift_score,
            feature_drift_scores=feature_drift_scores,
            drift_type=drift_type,
            recommended_action=recommended_action
        )
    
    def _determine_drift_type(self, feature_scores: Dict[str, float]) -> str:
        """ドリフトタイプ判定"""
        # 簡略化した判定ロジック
        high_drift_features = [k for k, v in feature_scores.items() if v > 0.95]
        
        if len(high_drift_features) > len(feature_scores) * 0.5:
            return "covariate"  # 入力分布の大幅な変化
        elif len(high_drift_features) > 0:
            return "prior"  # 一部特徴量の分布変化
        else:
            return "concept"  # 入力と出力の関係性変化の可能性
    
    def _recommend_action(self, drift_detected: bool, 
                          drift_score: float, 
                          drift_type: str) -> str:
        """推奨アクション決定"""
        if not drift_detected:
            return "No action required"
        
        if drift_score > 0.8:
            if drift_type == "concept":
                return "Immediate model retraining required"
            else:
                return "Investigate data pipeline and consider retraining"
        elif drift_score > 0.5:
            return "Schedule model retraining within 1 week"
        else:
            return "Monitor closely, consider retraining if drift persists"

class ModelPerformanceMonitor:
    """モデル性能監視"""
    def __init__(self, baseline_metrics: Dict[str, float]):
        self.baseline_metrics = baseline_metrics
        self.performance_history = []
        self.alert_thresholds = {
            'accuracy': 0.05,  # 5%以上の精度低下
            'precision': 0.05,
            'recall': 0.05,
            'f1': 0.05,
            'auc': 0.03
        }
    
    def evaluate_performance(self, y_true, y_pred, y_prob=None) -> Dict:
        """性能評価"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_prob is not None:
            metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        metrics['timestamp'] = datetime.now().isoformat()
        
        self.performance_history.append(metrics)
        
        return metrics
    
    def check_degradation(self, current_metrics: Dict) -> List[str]:
        """性能劣化チェック"""
        alerts = []
        
        for metric_name, baseline_value in self.baseline_metrics.items():
            if metric_name in current_metrics:
                current_value = current_metrics[metric_name]
                degradation = baseline_value - current_value
                
                if metric_name in self.alert_thresholds:
                    if degradation > self.alert_thresholds[metric_name]:
                        alerts.append(
                            f"{metric_name} degraded by {degradation:.3f} "
                            f"(baseline: {baseline_value:.3f}, current: {current_value:.3f})"
                        )
        
        return alerts
    
    def get_performance_trend(self, window: int = 10) -> Dict:
        """性能トレンド分析"""
        if len(self.performance_history) < window:
            return {}
        
        recent_history = self.performance_history[-window:]
        trends = {}
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            values = [h[metric] for h in recent_history if metric in h]
            if values:
                # 線形回帰でトレンド計算
                x = np.arange(len(values))
                slope, intercept = np.polyfit(x, values, 1)
                trends[metric] = {
                    'slope': slope,
                    'direction': 'improving' if slope > 0 else 'degrading',
                    'current': values[-1],
                    'average': np.mean(values)
                }
        
        return trends
```

---

## 3. A/Bテストとカナリアデプロイメント

### 実験プラットフォーム

#### 1. A/Bテストフレームワーク
```python
import hashlib
from typing import Dict, List, Optional, Tuple
import json
import redis
from dataclasses import dataclass, asdict
import random

@dataclass
class Experiment:
    id: str
    name: str
    status: str  # "draft", "running", "paused", "completed"
    variants: List[Dict]  # [{"name": "control", "weight": 50}, {"name": "treatment", "weight": 50}]
    targeting_rules: Optional[Dict]
    success_metrics: List[str]
    start_date: Optional[str]
    end_date: Optional[str]
    
@dataclass
class Assignment:
    user_id: str
    experiment_id: str
    variant: str
    timestamp: str

class ABTestingPlatform:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.experiments = {}
        
    def create_experiment(self, experiment: Experiment) -> str:
        """実験作成"""
        # 実験情報をRedisに保存
        exp_key = f"experiment:{experiment.id}"
        self.redis.hset(exp_key, mapping=asdict(experiment))
        
        # アクティブ実験リストに追加
        if experiment.status == "running":
            self.redis.sadd("active_experiments", experiment.id)
        
        return experiment.id
    
    def get_variant(self, user_id: str, experiment_id: str) -> str:
        """ユーザーのバリアント割り当て取得"""
        # 既存の割り当てチェック
        assignment_key = f"assignment:{experiment_id}:{user_id}"
        existing_assignment = self.redis.get(assignment_key)
        
        if existing_assignment:
            return existing_assignment.decode('utf-8')
        
        # 新規割り当て
        experiment = self.get_experiment(experiment_id)
        if not experiment or experiment['status'] != 'running':
            return "control"  # デフォルト
        
        # ターゲティングルールチェック
        if not self._check_targeting(user_id, experiment.get('targeting_rules')):
            return "control"
        
        # 決定論的割り当て（ハッシュベース）
        variant = self._assign_variant(user_id, experiment_id, experiment['variants'])
        
        # 割り当て保存
        self.redis.setex(
            assignment_key, 
            86400 * 30,  # 30日間保持
            variant
        )
        
        # 割り当てログ記録
        self._log_assignment(Assignment(
            user_id=user_id,
            experiment_id=experiment_id,
            variant=variant,
            timestamp=datetime.now().isoformat()
        ))
        
        return variant
    
    def _assign_variant(self, user_id: str, experiment_id: str, 
                       variants: List[Dict]) -> str:
        """決定論的バリアント割り当て"""
        # ユーザーIDと実験IDのハッシュ値計算
        hash_input = f"{user_id}:{experiment_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # ハッシュ値を0-100の範囲にマップ
        bucket = hash_value % 100
        
        # 重み付けに基づいてバリアント決定
        cumulative_weight = 0
        for variant in variants:
            cumulative_weight += variant['weight']
            if bucket < cumulative_weight:
                return variant['name']
        
        return variants[-1]['name']  # フォールバック
    
    def record_conversion(self, user_id: str, experiment_id: str, 
                         metric: str, value: float = 1.0):
        """コンバージョン記録"""
        # バリアント取得
        variant = self.get_variant(user_id, experiment_id)
        
        # メトリクス記録
        metric_key = f"metrics:{experiment_id}:{variant}:{metric}"
        self.redis.hincrby(metric_key, "count", 1)
        self.redis.hincrbyfloat(metric_key, "sum", value)
        
        # ユニークユーザー記録
        unique_key = f"unique:{experiment_id}:{variant}:{metric}"
        self.redis.sadd(unique_key, user_id)
    
    def get_experiment_results(self, experiment_id: str) -> Dict:
        """実験結果取得"""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return {}
        
        results = {
            'experiment_id': experiment_id,
            'variants': {}
        }
        
        for variant in experiment['variants']:
            variant_name = variant['name']
            variant_results = {
                'users': self.redis.scard(f"assignment:{experiment_id}:{variant_name}"),
                'metrics': {}
            }
            
            for metric in experiment['success_metrics']:
                metric_key = f"metrics:{experiment_id}:{variant_name}:{metric}"
                count = int(self.redis.hget(metric_key, "count") or 0)
                sum_value = float(self.redis.hget(metric_key, "sum") or 0)
                unique_users = self.redis.scard(f"unique:{experiment_id}:{variant_name}:{metric}")
                
                variant_results['metrics'][metric] = {
                    'count': count,
                    'sum': sum_value,
                    'average': sum_value / count if count > 0 else 0,
                    'unique_users': unique_users,
                    'conversion_rate': unique_users / variant_results['users'] 
                                      if variant_results['users'] > 0 else 0
                }
            
            results['variants'][variant_name] = variant_results
        
        # 統計的有意性計算
        results['statistical_significance'] = self._calculate_significance(results)
        
        return results
    
    def _calculate_significance(self, results: Dict) -> Dict:
        """統計的有意性計算（簡略版）"""
        from scipy import stats
        
        significance_results = {}
        variants = list(results['variants'].keys())
        
        if len(variants) != 2:
            return significance_results
        
        control = results['variants'][variants[0]]
        treatment = results['variants'][variants[1]]
        
        for metric in control.get('metrics', {}):
            control_conversions = control['metrics'][metric]['unique_users']
            control_total = control['users']
            treatment_conversions = treatment['metrics'][metric]['unique_users']
            treatment_total = treatment['users']
            
            if control_total > 0 and treatment_total > 0:
                # 二項検定
                control_rate = control_conversions / control_total
                treatment_rate = treatment_conversions / treatment_total
                
                pooled_rate = (control_conversions + treatment_conversions) / \
                             (control_total + treatment_total)
                
                se = np.sqrt(pooled_rate * (1 - pooled_rate) * 
                           (1/control_total + 1/treatment_total))
                
                if se > 0:
                    z_score = (treatment_rate - control_rate) / se
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                    
                    significance_results[metric] = {
                        'control_rate': control_rate,
                        'treatment_rate': treatment_rate,
                        'relative_improvement': (treatment_rate - control_rate) / control_rate 
                                              if control_rate > 0 else 0,
                        'p_value': p_value,
                        'is_significant': p_value < 0.05
                    }
        
        return significance_results
```

#### 2. カナリアデプロイメント
```python
class CanaryDeployment:
    def __init__(self, initial_percentage: float = 5.0):
        self.initial_percentage = initial_percentage
        self.current_percentage = initial_percentage
        self.deployment_id = None
        self.start_time = None
        self.metrics_history = []
        
    def start_deployment(self, deployment_id: str, new_model_version: str):
        """カナリアデプロイメント開始"""
        self.deployment_id = deployment_id
        self.new_model_version = new_model_version
        self.start_time = datetime.now()
        self.current_percentage = self.initial_percentage
        
        # トラフィック設定更新
        self._update_traffic_split(self.current_percentage)
        
        # 監視開始
        self._start_monitoring()
        
        return {
            "deployment_id": deployment_id,
            "status": "started",
            "traffic_percentage": self.current_percentage
        }
    
    def _update_traffic_split(self, percentage: float):
        """トラフィック分割更新"""
        # Kubernetes Ingress or Service Mesh設定更新
        config = {
            "apiVersion": "networking.istio.io/v1alpha3",
            "kind": "VirtualService",
            "metadata": {
                "name": "ml-service"
            },
            "spec": {
                "http": [{
                    "match": [{"headers": {"canary": {"exact": "true"}}}],
                    "route": [
                        {
                            "destination": {
                                "host": "ml-service",
                                "subset": "canary"
                            },
                            "weight": int(percentage)
                        },
                        {
                            "destination": {
                                "host": "ml-service",
                                "subset": "stable"
                            },
                            "weight": int(100 - percentage)
                        }
                    ]
                }]
            }
        }
        
        # Apply configuration (simplified)
        # kubectl_apply(config)
        
    def evaluate_canary(self) -> Dict:
        """カナリー評価"""
        metrics = self._collect_metrics()
        
        evaluation = {
            "deployment_id": self.deployment_id,
            "current_percentage": self.current_percentage,
            "duration_minutes": (datetime.now() - self.start_time).seconds / 60,
            "metrics": metrics,
            "health_check": self._health_check(metrics),
            "recommendation": None
        }
        
        # 推奨アクション決定
        if evaluation["health_check"]["passed"]:
            if self.current_percentage < 100:
                evaluation["recommendation"] = "increase_traffic"
            else:
                evaluation["recommendation"] = "complete_deployment"
        else:
            evaluation["recommendation"] = "rollback"
        
        return evaluation
    
    def _health_check(self, metrics: Dict) -> Dict:
        """ヘルスチェック"""
        checks = {
            "error_rate": metrics["error_rate"] < 0.01,
            "p99_latency": metrics["p99_latency"] < 500,
            "accuracy_degradation": metrics.get("accuracy_delta", 0) > -0.02
        }
        
        return {
            "passed": all(checks.values()),
            "checks": checks
        }
    
    def progressive_rollout(self):
        """段階的ロールアウト"""
        evaluation = self.evaluate_canary()
        
        if evaluation["recommendation"] == "increase_traffic":
            # トラフィック増加スケジュール
            if self.current_percentage < 10:
                self.current_percentage = 10
            elif self.current_percentage < 25:
                self.current_percentage = 25
            elif self.current_percentage < 50:
                self.current_percentage = 50
            elif self.current_percentage < 100:
                self.current_percentage = 100
            
            self._update_traffic_split(self.current_percentage)
            
            return {
                "action": "increased_traffic",
                "new_percentage": self.current_percentage
            }
            
        elif evaluation["recommendation"] == "rollback":
            return self.rollback()
        
        elif evaluation["recommendation"] == "complete_deployment":
            return self.complete_deployment()
    
    def rollback(self):
        """ロールバック"""
        self._update_traffic_split(0)
        
        return {
            "action": "rollback",
            "reason": "Health check failed",
            "deployment_id": self.deployment_id
        }
    
    def complete_deployment(self):
        """デプロイメント完了"""
        self._update_traffic_split(100)
        
        # 旧バージョンの削除スケジュール
        self._schedule_old_version_removal()
        
        return {
            "action": "completed",
            "deployment_id": self.deployment_id,
            "duration_minutes": (datetime.now() - self.start_time).seconds / 60
        }
```

---

## 4. 障害対応とインシデント管理

### インシデント対応フレームワーク

#### 1. 自動障害検知と対応
```python
class IncidentManager:
    def __init__(self):
        self.incidents = []
        self.runbooks = self._load_runbooks()
        self.on_call_schedule = self._load_on_call_schedule()
        
    def _load_runbooks(self) -> Dict:
        """ランブック読み込み"""
        return {
            "high_latency": {
                "detection": "p99_latency > 1000ms",
                "severity": "P2",
                "auto_remediation": [
                    "scale_up_replicas",
                    "clear_cache",
                    "restart_unhealthy_pods"
                ],
                "manual_steps": [
                    "Check database connection pool",
                    "Review recent deployments",
                    "Analyze slow query logs"
                ],
                "escalation": "15 minutes"
            },
            "model_drift": {
                "detection": "drift_score > 0.8",
                "severity": "P3",
                "auto_remediation": [
                    "switch_to_fallback_model",
                    "trigger_retraining_pipeline"
                ],
                "manual_steps": [
                    "Analyze drift patterns",
                    "Review data pipeline",
                    "Validate training data quality"
                ],
                "escalation": "1 hour"
            },
            "service_down": {
                "detection": "health_check_failures > 3",
                "severity": "P1",
                "auto_remediation": [
                    "restart_service",
                    "failover_to_backup_region"
                ],
                "manual_steps": [
                    "Check infrastructure status",
                    "Review error logs",
                    "Coordinate with infrastructure team"
                ],
                "escalation": "5 minutes"
            }
        }
    
    def detect_incident(self, metrics: Dict) -> Optional[Dict]:
        """インシデント検出"""
        for incident_type, runbook in self.runbooks.items():
            if self._evaluate_condition(runbook["detection"], metrics):
                return self.create_incident(
                    incident_type=incident_type,
                    severity=runbook["severity"],
                    metrics=metrics
                )
        return None
    
    def create_incident(self, incident_type: str, severity: str, 
                        metrics: Dict) -> Dict:
        """インシデント作成"""
        incident = {
            "id": f"INC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "type": incident_type,
            "severity": severity,
            "status": "open",
            "created_at": datetime.now(),
            "assigned_to": self.get_on_call_engineer(),
            "metrics": metrics,
            "timeline": [],
            "resolution": None
        }
        
        self.incidents.append(incident)
        
        # 自動修復試行
        self._attempt_auto_remediation(incident)
        
        # 通知送信
        self._send_notifications(incident)
        
        return incident
    
    def _attempt_auto_remediation(self, incident: Dict):
        """自動修復試行"""
        runbook = self.runbooks.get(incident["type"])
        if not runbook:
            return
        
        for action in runbook["auto_remediation"]:
            try:
                result = self._execute_remediation_action(action)
                incident["timeline"].append({
                    "timestamp": datetime.now(),
                    "action": f"Auto-remediation: {action}",
                    "result": result
                })
                
                # 修復成功チェック
                if self._check_incident_resolved(incident):
                    incident["status"] = "resolved"
                    incident["resolution"] = f"Auto-resolved by {action}"
                    break
                    
            except Exception as e:
                incident["timeline"].append({
                    "timestamp": datetime.now(),
                    "action": f"Auto-remediation failed: {action}",
                    "error": str(e)
                })
    
    def _execute_remediation_action(self, action: str) -> Dict:
        """修復アクション実行"""
        actions = {
            "scale_up_replicas": lambda: self._scale_deployment(replicas=10),
            "clear_cache": lambda: self._clear_redis_cache(),
            "restart_unhealthy_pods": lambda: self._restart_pods(),
            "switch_to_fallback_model": lambda: self._switch_model("fallback"),
            "trigger_retraining_pipeline": lambda: self._trigger_pipeline("retrain"),
            "restart_service": lambda: self._restart_service(),
            "failover_to_backup_region": lambda: self._failover_region()
        }
        
        if action in actions:
            return actions[action]()
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def postmortem_analysis(self, incident_id: str) -> Dict:
        """ポストモーテム分析"""
        incident = self._get_incident(incident_id)
        
        analysis = {
            "incident_id": incident_id,
            "summary": {
                "duration": self._calculate_duration(incident),
                "severity": incident["severity"],
                "impact": self._assess_impact(incident)
            },
            "timeline": incident["timeline"],
            "root_cause": self._analyze_root_cause(incident),
            "contributing_factors": self._identify_contributing_factors(incident),
            "what_went_well": [],
            "what_went_wrong": [],
            "action_items": self._generate_action_items(incident)
        }
        
        return analysis
    
    def _generate_action_items(self, incident: Dict) -> List[Dict]:
        """アクションアイテム生成"""
        action_items = []
        
        # 共通のアクションアイテム
        action_items.append({
            "title": "Update runbook",
            "description": f"Update runbook for {incident['type']} based on learnings",
            "owner": "SRE Team",
            "due_date": (datetime.now() + timedelta(days=7)).isoformat()
        })
        
        # インシデントタイプ別のアクションアイテム
        if incident["type"] == "model_drift":
            action_items.append({
                "title": "Implement proactive drift monitoring",
                "description": "Set up alerts for early drift detection",
                "owner": "ML Team",
                "due_date": (datetime.now() + timedelta(days=14)).isoformat()
            })
        
        return action_items
```

---

## まとめ

### 本章で習得したスキル

1. **信頼性工学**: SLI/SLO/SLA定義とエラーバジェット管理
2. **モニタリング**: ドリフト検出と性能監視
3. **実験管理**: A/Bテストとカナリアデプロイメント
4. **障害対応**: インシデント管理と自動修復

### プロダクション運用のベストプラクティス

#### 可観測性の三本柱
- **Metrics**: Prometheus + Grafana
- **Logging**: ELK Stack / Fluentd
- **Tracing**: Jaeger / Zipkin

#### デプロイメント戦略
- **Blue-Green**: 即座の切り替え
- **Canary**: 段階的ロールアウト
- **Feature Flags**: 機能単位の制御
- **Shadow Mode**: 本番トラフィックでのテスト

#### 障害対応
- **Runbooks**: 手順書の整備
- **On-Call**: ローテーション体制
- **Postmortem**: 障害から学ぶ文化
- **Chaos Engineering**: 計画的障害注入

### チェックリスト

#### デプロイメント前
- [ ] 単体テスト・統合テスト完了
- [ ] 性能テスト実施
- [ ] セキュリティスキャン完了
- [ ] ロールバック計画準備

#### デプロイメント中
- [ ] カナリアデプロイメント開始
- [ ] メトリクス監視
- [ ] エラー率チェック
- [ ] 段階的トラフィック増加

#### デプロイメント後
- [ ] 全メトリクス正常確認
- [ ] アラート設定確認
- [ ] ドキュメント更新
- [ ] 振り返り実施

---

## 参考資料
- [Google SRE Book](https://sre.google/sre-book/table-of-contents/)
- [ML Ops: Continuous delivery and automation](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Monitoring Machine Learning Models in Production](https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/)
- [The ML Test Score: A Rubric for ML Production Readiness](https://research.google/pubs/pub46555/)