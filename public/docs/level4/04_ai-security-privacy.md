# AIセキュリティとプライバシー

![AIセキュリティ](/images/illustrations/level4-security.jpg)

## 学習目標
- AIシステムのセキュリティ脅威と対策を理解する
- プライバシー保護技術（差分プライバシー、連合学習等）を実装する
- AIモデルへの攻撃と防御手法を習得する
- データガバナンスとコンプライアンス対応を構築する

## 想定学習時間
約10-12時間（実装演習・セキュリティ監査演習含む）

---

## 1. AIシステムのセキュリティ脅威

### 攻撃ベクトルと脅威モデル

#### 1. 敵対的攻撃（Adversarial Attacks）
```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class AdversarialAttacks:
    """敵対的攻撃の実装"""
    
    @staticmethod
    def fgsm_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, 
                    epsilon: float = 0.3) -> torch.Tensor:
        """Fast Gradient Sign Method (FGSM)攻撃"""
        x.requires_grad = True
        
        # 順伝播
        outputs = model(x)
        loss = F.cross_entropy(outputs, y)
        
        # 逆伝播
        model.zero_grad()
        loss.backward()
        
        # 敵対的摂動の生成
        data_grad = x.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_x = x + epsilon * sign_data_grad
        
        # クリッピング（値の範囲を維持）
        perturbed_x = torch.clamp(perturbed_x, 0, 1)
        
        return perturbed_x
    
    @staticmethod
    def pgd_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                   epsilon: float = 0.3, alpha: float = 0.01, 
                   num_iter: int = 40) -> torch.Tensor:
        """Projected Gradient Descent (PGD)攻撃"""
        perturbed_x = x.clone().detach()
        perturbed_x = perturbed_x + torch.empty_like(perturbed_x).uniform_(-epsilon, epsilon)
        perturbed_x = torch.clamp(perturbed_x, 0, 1).detach()
        
        for _ in range(num_iter):
            perturbed_x.requires_grad = True
            outputs = model(perturbed_x)
            loss = F.cross_entropy(outputs, y)
            
            model.zero_grad()
            loss.backward()
            
            adv_x = perturbed_x + alpha * perturbed_x.grad.sign()
            eta = torch.clamp(adv_x - x, min=-epsilon, max=epsilon)
            perturbed_x = torch.clamp(x + eta, min=0, max=1).detach()
        
        return perturbed_x
    
    @staticmethod
    def carlini_wagner_attack(model: nn.Module, x: torch.Tensor, 
                             y: torch.Tensor, c: float = 1.0, 
                             kappa: float = 0, max_iter: int = 1000) -> torch.Tensor:
        """Carlini & Wagner (C&W)攻撃"""
        # L2ノルムベースの最適化攻撃
        batch_size = x.shape[0]
        
        # 変数の初期化
        w = torch.zeros_like(x, requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=0.01)
        
        best_adv = x.clone()
        best_dist = torch.full((batch_size,), float('inf'))
        
        for iteration in range(max_iter):
            # tanh空間での変換
            adv_x = 0.5 * (torch.tanh(w) + 1)
            
            # 距離計算
            dist = torch.norm((adv_x - x).view(batch_size, -1), p=2, dim=1)
            
            # 分類損失
            outputs = model(adv_x)
            one_hot_y = F.one_hot(y, outputs.shape[1]).float()
            
            real = torch.sum(outputs * one_hot_y, dim=1)
            other = torch.max(outputs * (1 - one_hot_y) - 1e10 * one_hot_y, dim=1)[0]
            
            loss_cls = torch.clamp(real - other + kappa, min=0)
            loss = torch.sum(dist) + c * torch.sum(loss_cls)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 最良の敵対的サンプルを保存
            pred = outputs.argmax(dim=1)
            mask = (pred != y) & (dist < best_dist)
            best_dist[mask] = dist[mask]
            best_adv[mask] = adv_x[mask]
        
        return best_adv

class AdversarialDefense:
    """敵対的防御メカニズム"""
    
    @staticmethod
    def adversarial_training(model: nn.Module, train_loader, 
                           epsilon: float = 0.3, epochs: int = 10):
        """敵対的訓練"""
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        attack = AdversarialAttacks()
        
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                # 通常のサンプルで訓練
                optimizer.zero_grad()
                output = model(data)
                loss_clean = F.cross_entropy(output, target)
                
                # 敵対的サンプル生成
                adv_data = attack.fgsm_attack(model, data, target, epsilon)
                
                # 敵対的サンプルで訓練
                output_adv = model(adv_data)
                loss_adv = F.cross_entropy(output_adv, target)
                
                # 総合損失
                loss = 0.5 * loss_clean + 0.5 * loss_adv
                loss.backward()
                optimizer.step()
        
        return model
    
    @staticmethod
    def input_preprocessing(x: torch.Tensor, method: str = 'jpeg') -> torch.Tensor:
        """入力前処理による防御"""
        if method == 'jpeg':
            # JPEG圧縮シミュレーション
            from torchvision import transforms
            from PIL import Image
            
            # テンソルを画像に変換
            to_pil = transforms.ToPILImage()
            to_tensor = transforms.ToTensor()
            
            processed = []
            for img_tensor in x:
                img = to_pil(img_tensor.cpu())
                # JPEG圧縮
                import io
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=75)
                buffer.seek(0)
                img = Image.open(buffer)
                processed.append(to_tensor(img))
            
            return torch.stack(processed).to(x.device)
        
        elif method == 'bit_depth_reduction':
            # ビット深度削減
            levels = 32
            x_reduced = torch.round(x * levels) / levels
            return x_reduced
        
        elif method == 'gaussian_noise':
            # ガウシアンノイズ追加
            noise = torch.randn_like(x) * 0.01
            return torch.clamp(x + noise, 0, 1)
        
        return x
    
    @staticmethod
    def certified_defense(model: nn.Module, x: torch.Tensor, 
                         sigma: float = 0.25, n_samples: int = 100,
                         alpha: float = 0.001) -> Tuple[torch.Tensor, torch.Tensor]:
        """認証済み防御（Randomized Smoothing）"""
        batch_size = x.shape[0]
        num_classes = 10  # 仮定
        
        counts = torch.zeros(batch_size, num_classes)
        
        for _ in range(n_samples):
            noise = torch.randn_like(x) * sigma
            noisy_x = x + noise
            with torch.no_grad():
                outputs = model(noisy_x)
                predictions = outputs.argmax(dim=1)
                for i in range(batch_size):
                    counts[i, predictions[i]] += 1
        
        # 最も頻度の高いクラスとその確率
        top_counts, top_classes = counts.max(dim=1)
        confidence = top_counts / n_samples
        
        # 認証半径の計算
        from scipy.stats import norm
        certified_radius = sigma * norm.ppf(confidence.cpu().numpy())
        certified_radius = torch.tensor(certified_radius)
        
        return top_classes, certified_radius
```

#### 2. データポイズニング攻撃と防御
```python
class DataPoisoning:
    """データポイズニング攻撃と防御"""
    
    @staticmethod
    def backdoor_attack(x_train: np.ndarray, y_train: np.ndarray, 
                        poison_rate: float = 0.1, 
                        target_label: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """バックドア攻撃の実装"""
        n_samples = len(x_train)
        n_poison = int(n_samples * poison_rate)
        
        # ポイズニングするサンプルのインデックス
        poison_indices = np.random.choice(n_samples, n_poison, replace=False)
        
        x_poisoned = x_train.copy()
        y_poisoned = y_train.copy()
        
        # トリガーパターンの追加（例：画像の隅に小さなパッチ）
        trigger_pattern = np.ones((3, 3, x_train.shape[-1])) * 255
        
        for idx in poison_indices:
            # トリガーを追加
            x_poisoned[idx, -3:, -3:] = trigger_pattern
            # ラベルを変更
            y_poisoned[idx] = target_label
        
        return x_poisoned, y_poisoned
    
    @staticmethod
    def detect_poisoned_samples(model, x_train, y_train, 
                               threshold: float = 2.0) -> np.ndarray:
        """ポイズニングされたサンプルの検出"""
        from sklearn.ensemble import IsolationForest
        
        # 特徴抽出（最終層の前の層の出力を使用）
        features = []
        with torch.no_grad():
            for x in x_train:
                # モデルの中間層の出力を取得
                feature = model.get_features(x)  # 仮定：特徴抽出メソッド
                features.append(feature.numpy())
        
        features = np.array(features)
        
        # Isolation Forestで異常検知
        clf = IsolationForest(contamination='auto', random_state=42)
        predictions = clf.fit_predict(features)
        
        # 外れ値として検出されたサンプル
        poisoned_indices = np.where(predictions == -1)[0]
        
        return poisoned_indices
    
    @staticmethod
    def data_sanitization(x_train, y_train, model, 
                         validation_split: float = 0.1):
        """データサニタイゼーション"""
        from sklearn.model_selection import train_test_split
        
        # 検証用データの分割
        x_clean, x_val, y_clean, y_val = train_test_split(
            x_train, y_train, test_size=validation_split, stratify=y_train
        )
        
        # 影響度スコアの計算
        influence_scores = []
        
        for i in range(len(x_train)):
            # Leave-one-out訓練
            x_loo = np.concatenate([x_train[:i], x_train[i+1:]])
            y_loo = np.concatenate([y_train[:i], y_train[i+1:]])
            
            # 簡略化：実際には再訓練が必要
            # ここでは影響度の近似値を計算
            with torch.no_grad():
                pred_with = model(torch.tensor(x_train[i:i+1]))
                pred_without = model(torch.tensor(x_val))
                
                influence = np.abs(pred_with.numpy() - pred_without.mean(0).numpy()).sum()
                influence_scores.append(influence)
        
        influence_scores = np.array(influence_scores)
        
        # 高影響度のサンプルを除外
        threshold = np.percentile(influence_scores, 95)
        clean_indices = influence_scores < threshold
        
        return x_train[clean_indices], y_train[clean_indices]
```

---

## 2. プライバシー保護技術

### 差分プライバシー（Differential Privacy）

#### 1. 差分プライバシー機構の実装
```python
class DifferentialPrivacy:
    """差分プライバシー実装"""
    
    @staticmethod
    def laplace_mechanism(value: float, sensitivity: float, 
                         epsilon: float) -> float:
        """ラプラス機構"""
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    @staticmethod
    def gaussian_mechanism(value: float, sensitivity: float, 
                          epsilon: float, delta: float) -> float:
        """ガウス機構"""
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        noise = np.random.normal(0, sigma)
        return value + noise
    
    @staticmethod
    def exponential_mechanism(scores: np.ndarray, sensitivity: float, 
                             epsilon: float) -> int:
        """指数機構（離散値の選択）"""
        probabilities = np.exp(epsilon * scores / (2 * sensitivity))
        probabilities /= probabilities.sum()
        return np.random.choice(len(scores), p=probabilities)

class DPSGDOptimizer(torch.optim.Optimizer):
    """差分プライバシーSGD最適化"""
    
    def __init__(self, params, lr=0.01, max_grad_norm=1.0, 
                 noise_multiplier=1.0, batch_size=32):
        defaults = dict(lr=lr, max_grad_norm=max_grad_norm,
                       noise_multiplier=noise_multiplier, 
                       batch_size=batch_size)
        super(DPSGDOptimizer, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """差分プライバシー付きパラメータ更新"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            max_grad_norm = group['max_grad_norm']
            noise_multiplier = group['noise_multiplier']
            
            # Per-sample gradient clipping
            total_norm = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            clip_coef = max_grad_norm / (total_norm + 1e-6)
            if clip_coef < 1:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.data.mul_(clip_coef)
            
            # ガウシアンノイズの追加
            for p in group['params']:
                if p.grad is None:
                    continue
                
                noise_std = max_grad_norm * noise_multiplier
                noise = torch.normal(0, noise_std, size=p.grad.shape,
                                   device=p.grad.device)
                p.grad.data.add_(noise / group['batch_size'])
                
                # パラメータ更新
                p.data.add_(p.grad.data, alpha=-group['lr'])
        
        return loss

class PrivacyAccountant:
    """プライバシー会計"""
    
    def __init__(self, delta: float = 1e-5):
        self.delta = delta
        self.epsilon_history = []
        
    def compute_epsilon(self, steps: int, noise_multiplier: float, 
                       sample_rate: float) -> float:
        """RDPを使用したεの計算"""
        from scipy import special
        
        # Rényi Differential Privacy (RDP)の計算
        orders = [1 + x / 10.0 for x in range(1, 100)]
        rdp = np.zeros(len(orders))
        
        for i, order in enumerate(orders):
            rdp[i] = self._compute_rdp(
                sample_rate, noise_multiplier, steps, order
            )
        
        # RDPから(ε,δ)-DPへの変換
        eps, opt_order = self._rdp_to_dp(orders, rdp, self.delta)
        
        self.epsilon_history.append(eps)
        return eps
    
    def _compute_rdp(self, q: float, sigma: float, steps: int, 
                    alpha: float) -> float:
        """RDPの計算"""
        if alpha == 1:
            return 0
        
        # Gaussian mechanism
        if np.isinf(alpha):
            return q * steps * (1 / (2 * sigma**2))
        
        return q * steps * alpha / (2 * sigma**2)
    
    def _rdp_to_dp(self, orders: list, rdp: np.ndarray, 
                   delta: float) -> Tuple[float, float]:
        """RDPから(ε,δ)-DPへの変換"""
        eps_list = []
        
        for i, order in enumerate(orders):
            if order == 1:
                continue
            eps = rdp[i] + np.log(1/delta) / (order - 1)
            eps_list.append(eps)
        
        min_idx = np.argmin(eps_list)
        return eps_list[min_idx], orders[min_idx + 1]
```

### 連合学習のプライバシー

#### 1. セキュア集約（Secure Aggregation）
```python
class SecureAggregation:
    """セキュアな連合学習集約"""
    
    def __init__(self, num_clients: int, threshold: int):
        self.num_clients = num_clients
        self.threshold = threshold  # 最小参加クライアント数
        self.client_keys = {}
        
    def setup_keys(self):
        """鍵ペアのセットアップ"""
        from cryptography.hazmat.primitives import serialization, hashes
        from cryptography.hazmat.primitives.asymmetric import rsa, padding
        
        for client_id in range(self.num_clients):
            # RSA鍵ペア生成
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            public_key = private_key.public_key()
            
            self.client_keys[client_id] = {
                'private': private_key,
                'public': public_key
            }
    
    def generate_masks(self, client_id: int) -> Dict[int, np.ndarray]:
        """ペアワイズマスクの生成"""
        masks = {}
        
        for other_id in range(self.num_clients):
            if other_id == client_id:
                continue
            
            # 共有シードから決定論的な乱数生成
            shared_seed = min(client_id, other_id) * 1000 + max(client_id, other_id)
            np.random.seed(shared_seed)
            
            if client_id < other_id:
                mask = np.random.randn(100)  # モデルサイズに応じて調整
            else:
                mask = -np.random.randn(100)
            
            masks[other_id] = mask
        
        return masks
    
    def secure_aggregate(self, client_updates: Dict[int, np.ndarray]) -> np.ndarray:
        """セキュア集約の実行"""
        if len(client_updates) < self.threshold:
            raise ValueError(f"Not enough clients: {len(client_updates)} < {self.threshold}")
        
        # マスク付き更新の収集
        masked_updates = {}
        
        for client_id, update in client_updates.items():
            masks = self.generate_masks(client_id)
            masked_update = update.copy()
            
            for other_id, mask in masks.items():
                if other_id in client_updates:  # 参加しているクライアントのみ
                    masked_update += mask
            
            masked_updates[client_id] = masked_update
        
        # 集約（マスクは相殺される）
        aggregated = np.zeros_like(list(masked_updates.values())[0])
        for masked_update in masked_updates.values():
            aggregated += masked_update
        
        # 平均化
        aggregated /= len(client_updates)
        
        return aggregated

class HomomorphicEncryption:
    """準同型暗号を使用した連合学習"""
    
    def __init__(self):
        import tenseal as ts
        
        # TenSEALコンテキスト設定
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.generate_galois_keys()
        context.global_scale = 2**40
        self.context = context
    
    def encrypt_weights(self, weights: np.ndarray) -> 'CKKSVector':
        """重みの暗号化"""
        import tenseal as ts
        
        encrypted = ts.ckks_vector(self.context, weights.flatten())
        return encrypted
    
    def aggregate_encrypted(self, encrypted_updates: list) -> 'CKKSVector':
        """暗号化された状態での集約"""
        result = encrypted_updates[0]
        for update in encrypted_updates[1:]:
            result += update
        
        # 平均化
        result *= 1.0 / len(encrypted_updates)
        
        return result
    
    def decrypt_weights(self, encrypted_weights) -> np.ndarray:
        """重みの復号化"""
        decrypted = encrypted_weights.decrypt()
        return np.array(decrypted)
```

---

## 3. モデル保護とアクセス制御

### モデル抽出攻撃への防御

#### 1. モデル抽出攻撃の検出と防御
```python
class ModelProtection:
    """モデル保護機構"""
    
    def __init__(self, model, rate_limit: int = 1000):
        self.model = model
        self.rate_limit = rate_limit
        self.query_history = []
        self.suspicious_users = set()
        
    def detect_extraction_attack(self, queries: list) -> bool:
        """モデル抽出攻撃の検出"""
        # 1. クエリレートの異常検知
        query_rate = len(queries) / (time.time() - queries[0]['timestamp'])
        if query_rate > self.rate_limit:
            return True
        
        # 2. クエリパターンの分析
        if self._detect_systematic_queries(queries):
            return True
        
        # 3. エントロピーベースの検出
        if self._detect_low_entropy_queries(queries):
            return True
        
        return False
    
    def _detect_systematic_queries(self, queries: list) -> bool:
        """系統的なクエリパターンの検出"""
        # 入力の分散を計算
        inputs = [q['input'] for q in queries]
        variances = np.var(inputs, axis=0)
        
        # 非常に規則的なパターン（グリッドサーチ等）の検出
        if np.all(variances < 0.01):
            return True
        
        return False
    
    def _detect_low_entropy_queries(self, queries: list) -> bool:
        """低エントロピークエリの検出"""
        from scipy.stats import entropy
        
        inputs = np.array([q['input'] for q in queries])
        
        # 各特徴量のエントロピー計算
        entropies = []
        for feature in inputs.T:
            hist, _ = np.histogram(feature, bins=10)
            entropies.append(entropy(hist + 1e-10))
        
        # 低エントロピーの特徴量が多い場合
        if sum(e < 0.5 for e in entropies) > len(entropies) * 0.7:
            return True
        
        return False
    
    def protected_inference(self, input_data, user_id: str):
        """保護された推論"""
        # レート制限チェック
        if user_id in self.suspicious_users:
            raise PermissionError(f"User {user_id} is blocked")
        
        # クエリ記録
        self.query_history.append({
            'user_id': user_id,
            'input': input_data,
            'timestamp': time.time()
        })
        
        # 攻撃検出
        user_queries = [q for q in self.query_history if q['user_id'] == user_id]
        if len(user_queries) > 100:
            if self.detect_extraction_attack(user_queries[-100:]):
                self.suspicious_users.add(user_id)
                raise PermissionError(f"Suspicious activity detected for user {user_id}")
        
        # 出力の摂動
        output = self.model(input_data)
        protected_output = self._add_output_perturbation(output)
        
        return protected_output
    
    def _add_output_perturbation(self, output: np.ndarray, 
                                epsilon: float = 0.01) -> np.ndarray:
        """出力への摂動追加"""
        # 確率的丸め
        if len(output.shape) == 1:  # 分類の信頼度
            # Top-kの信頼度のみ返す
            k = 3
            top_k_indices = np.argsort(output)[-k:]
            protected_output = np.zeros_like(output)
            protected_output[top_k_indices] = output[top_k_indices]
            
            # ノイズ追加
            noise = np.random.laplace(0, epsilon, size=k)
            protected_output[top_k_indices] += noise
            
            # 正規化
            protected_output = np.clip(protected_output, 0, 1)
            protected_output /= protected_output.sum()
        else:
            # 回帰出力
            noise = np.random.laplace(0, epsilon, size=output.shape)
            protected_output = output + noise
        
        return protected_output

class ModelWatermarking:
    """モデル電子透かし"""
    
    def __init__(self, model, watermark_size: int = 100):
        self.model = model
        self.watermark_size = watermark_size
        self.watermark_key = None
        self.trigger_set = None
        
    def embed_watermark(self, trigger_ratio: float = 0.01):
        """電子透かしの埋め込み"""
        # トリガーセットの生成
        self.trigger_set = self._generate_trigger_set()
        
        # 透かしキーの生成
        self.watermark_key = np.random.randint(0, 2, self.watermark_size)
        
        # モデルの微調整でトリガーセットに対する出力を制御
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        
        for trigger_input, watermark_bit in zip(self.trigger_set, self.watermark_key):
            trigger_tensor = torch.tensor(trigger_input, dtype=torch.float32)
            
            # 特定の出力パターンを学習
            target_output = torch.zeros(self.model.output_size)
            target_output[watermark_bit] = 1.0
            
            for _ in range(10):  # 軽い微調整
                optimizer.zero_grad()
                output = self.model(trigger_tensor)
                loss = F.cross_entropy(output.unsqueeze(0), 
                                      target_output.argmax().unsqueeze(0))
                loss.backward()
                optimizer.step()
    
    def verify_watermark(self, suspect_model) -> float:
        """電子透かしの検証"""
        if self.trigger_set is None or self.watermark_key is None:
            raise ValueError("Watermark not embedded")
        
        matches = 0
        for trigger_input, expected_bit in zip(self.trigger_set, self.watermark_key):
            trigger_tensor = torch.tensor(trigger_input, dtype=torch.float32)
            
            with torch.no_grad():
                output = suspect_model(trigger_tensor)
                predicted = output.argmax().item()
                
                if predicted == expected_bit:
                    matches += 1
        
        verification_rate = matches / self.watermark_size
        
        # 統計的有意性の計算
        from scipy import stats
        p_value = stats.binom_test(matches, self.watermark_size, 0.5)
        
        return verification_rate, p_value
    
    def _generate_trigger_set(self) -> list:
        """トリガーセットの生成"""
        # アウトオブディストリビューションのサンプル生成
        trigger_set = []
        
        for _ in range(self.watermark_size):
            # ランダムノイズベースのトリガー
            trigger = np.random.randn(*self.model.input_shape) * 0.1
            trigger_set.append(trigger)
        
        return trigger_set
```

---

## 4. データガバナンスとコンプライアンス

### GDPR準拠のAIシステム

#### 1. プライバシー・バイ・デザイン実装
```python
class GDPRCompliantSystem:
    """GDPR準拠AIシステム"""
    
    def __init__(self):
        self.data_registry = {}
        self.consent_registry = {}
        self.audit_log = []
        
    def collect_data(self, user_id: str, data: dict, purpose: str) -> bool:
        """データ収集（同意確認付き）"""
        # 同意確認
        if not self.check_consent(user_id, purpose):
            return False
        
        # データ最小化
        minimized_data = self.minimize_data(data, purpose)
        
        # 仮名化
        pseudonymized_data = self.pseudonymize(minimized_data, user_id)
        
        # データ登録
        self.data_registry[user_id] = {
            'data': pseudonymized_data,
            'purpose': purpose,
            'collected_at': datetime.now(),
            'retention_period': self.get_retention_period(purpose)
        }
        
        # 監査ログ
        self.audit_log.append({
            'action': 'data_collection',
            'user_id': user_id,
            'purpose': purpose,
            'timestamp': datetime.now()
        })
        
        return True
    
    def minimize_data(self, data: dict, purpose: str) -> dict:
        """データ最小化"""
        required_fields = {
            'marketing': ['age_group', 'interests'],
            'analytics': ['usage_patterns', 'device_type'],
            'personalization': ['preferences', 'history']
        }
        
        if purpose in required_fields:
            minimized = {k: v for k, v in data.items() 
                        if k in required_fields[purpose]}
        else:
            minimized = data
        
        return minimized
    
    def pseudonymize(self, data: dict, user_id: str) -> dict:
        """仮名化"""
        import hashlib
        
        # ユーザーIDのハッシュ化
        pseudo_id = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        
        # 直接識別子の除去
        direct_identifiers = ['name', 'email', 'phone', 'address', 'ip_address']
        pseudonymized = {k: v for k, v in data.items() 
                        if k not in direct_identifiers}
        
        pseudonymized['pseudo_id'] = pseudo_id
        
        return pseudonymized
    
    def right_to_erasure(self, user_id: str) -> bool:
        """忘れられる権利の実行"""
        if user_id not in self.data_registry:
            return False
        
        # データ削除
        del self.data_registry[user_id]
        
        # モデルからの影響除去（Machine Unlearning）
        self.machine_unlearning(user_id)
        
        # 監査ログ
        self.audit_log.append({
            'action': 'data_erasure',
            'user_id': user_id,
            'timestamp': datetime.now()
        })
        
        return True
    
    def machine_unlearning(self, user_id: str):
        """機械的忘却（Machine Unlearning）"""
        # SISA (Sharded, Isolated, Sliced, and Aggregated) training
        # 影響を受けたシャードのみ再訓練
        
        # 簡略化された実装
        affected_shard = self.get_affected_shard(user_id)
        if affected_shard:
            self.retrain_shard(affected_shard, exclude_user=user_id)
    
    def data_portability(self, user_id: str) -> dict:
        """データポータビリティ権の実行"""
        if user_id not in self.data_registry:
            return {}
        
        user_data = self.data_registry[user_id]
        
        # 構造化された機械可読形式で提供
        portable_data = {
            'user_id': user_id,
            'data': user_data['data'],
            'metadata': {
                'collected_at': user_data['collected_at'].isoformat(),
                'purpose': user_data['purpose']
            }
        }
        
        return portable_data
    
    def privacy_impact_assessment(self, processing_activity: dict) -> dict:
        """プライバシー影響評価（PIA）"""
        assessment = {
            'activity': processing_activity,
            'risks': [],
            'mitigations': [],
            'residual_risk': None
        }
        
        # リスク評価
        if processing_activity.get('uses_ai', False):
            assessment['risks'].append({
                'type': 'algorithmic_bias',
                'severity': 'medium',
                'likelihood': 'possible'
            })
            assessment['mitigations'].append({
                'measure': 'bias_testing',
                'effectiveness': 'high'
            })
        
        if processing_activity.get('data_volume', 0) > 10000:
            assessment['risks'].append({
                'type': 'data_breach',
                'severity': 'high',
                'likelihood': 'unlikely'
            })
            assessment['mitigations'].append({
                'measure': 'encryption',
                'effectiveness': 'high'
            })
        
        # 残存リスクの計算
        assessment['residual_risk'] = self.calculate_residual_risk(assessment)
        
        return assessment
```

#### 2. 監査とコンプライアンス監視
```python
class ComplianceMonitor:
    """コンプライアンス監視システム"""
    
    def __init__(self):
        self.regulations = {
            'GDPR': GDPRCompliance(),
            'CCPA': CCPACompliance(),
            'HIPAA': HIPAACompliance()
        }
        self.violations = []
        
    def continuous_monitoring(self, system_logs: list) -> dict:
        """継続的コンプライアンス監視"""
        monitoring_results = {
            'timestamp': datetime.now(),
            'violations': [],
            'warnings': [],
            'compliance_score': 100.0
        }
        
        for log_entry in system_logs:
            # 各規制に対するチェック
            for regulation_name, regulation in self.regulations.items():
                result = regulation.check_compliance(log_entry)
                
                if result['violation']:
                    monitoring_results['violations'].append({
                        'regulation': regulation_name,
                        'violation_type': result['type'],
                        'severity': result['severity'],
                        'log_entry': log_entry
                    })
                    monitoring_results['compliance_score'] -= result['penalty']
                
                elif result.get('warning'):
                    monitoring_results['warnings'].append({
                        'regulation': regulation_name,
                        'warning_type': result['type'],
                        'log_entry': log_entry
                    })
        
        return monitoring_results
    
    def generate_audit_report(self, start_date: datetime, 
                            end_date: datetime) -> dict:
        """監査レポート生成"""
        report = {
            'period': f"{start_date} to {end_date}",
            'compliance_summary': {},
            'violations': [],
            'remediation_actions': [],
            'recommendations': []
        }
        
        # 規制ごとのコンプライアンス状況
        for regulation_name, regulation in self.regulations.items():
            compliance_status = regulation.get_compliance_status()
            report['compliance_summary'][regulation_name] = compliance_status
        
        # 違反の詳細
        report['violations'] = self.violations
        
        # 是正措置
        for violation in self.violations:
            remediation = self.get_remediation_action(violation)
            report['remediation_actions'].append(remediation)
        
        # 推奨事項
        report['recommendations'] = self.generate_recommendations()
        
        return report
    
    def automated_remediation(self, violation: dict) -> bool:
        """自動是正措置"""
        remediation_actions = {
            'excessive_data_retention': self.delete_expired_data,
            'missing_encryption': self.enable_encryption,
            'unauthorized_access': self.revoke_access,
            'missing_consent': self.request_consent
        }
        
        violation_type = violation['violation_type']
        if violation_type in remediation_actions:
            action = remediation_actions[violation_type]
            success = action(violation)
            
            if success:
                self.log_remediation(violation, violation_type)
            
            return success
        
        return False
```

---

## まとめ

### 本章で習得したスキル

1. **セキュリティ脅威対策**: 敵対的攻撃、データポイズニングへの防御
2. **プライバシー技術**: 差分プライバシー、連合学習、準同型暗号
3. **モデル保護**: 抽出攻撃防御、電子透かし
4. **コンプライアンス**: GDPR準拠、監査システム

### セキュリティベストプラクティス

#### 設計原則
- **Defense in Depth**: 多層防御
- **Least Privilege**: 最小権限の原則
- **Zero Trust**: ゼロトラスト
- **Privacy by Design**: プライバシー・バイ・デザイン

#### 実装チェックリスト

##### セキュリティ
- [ ] 入力検証とサニタイゼーション
- [ ] 敵対的訓練の実施
- [ ] モデル抽出攻撃対策
- [ ] アクセス制御とレート制限

##### プライバシー
- [ ] 差分プライバシーの適用
- [ ] データ最小化
- [ ] 仮名化・匿名化
- [ ] 同意管理

##### コンプライアンス
- [ ] GDPR/CCPA準拠
- [ ] 監査ログの記録
- [ ] データ保持ポリシー
- [ ] インシデント対応計画

### 継続的改善

1. **脅威モデリング**: 定期的な脅威評価
2. **セキュリティ監査**: 外部監査の実施
3. **インシデント演習**: 定期的な訓練
4. **最新動向の追跡**: 新しい攻撃手法への対応

---

## 参考資料
- [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- [TensorFlow Privacy](https://github.com/tensorflow/privacy)
- [PySyft - Federated Learning](https://github.com/OpenMined/PySyft)
- [AI Security & Privacy Guidelines](https://www.nist.gov/itl/ai-security-and-privacy)