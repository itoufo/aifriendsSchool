# カスタムAI開発・実装

![カスタムAI開発](/images/illustrations/level3-custom-ai.jpg)

## 学習目標
- GPTsやCustom GPTの設計・開発手法を習得する
- API連携による高度なAIシステム構築技術を身につける
- 機械学習モデルの基礎理解と実装スキルを向上させる
- セキュリティとプライバシーを考慮したAI開発手法を学ぶ

## 想定学習時間
約6-8時間（実装含む）

---

## 1. GPTs・Custom GPT開発

### GPTsの基本概念と活用範囲
GPTsは、特定の用途に特化したChatGPTのカスタムバージョンを作成する機能です。

#### GPTs開発の適用領域
- **業務特化型アシスタント**: 法務、経理、営業支援等
- **教育・トレーニング**: 社内研修、スキル開発支援
- **クリエイティブ支援**: コンテンツ制作、デザイン提案
- **分析・レポート**: データ分析、市場調査支援

### 効果的なGPTs設計手法

#### 1. 要件定義・ペルソナ設計
```markdown
【GPTs設計シート例】

■ 基本情報
- 名称: 営業提案書作成アシスタント
- 対象ユーザー: 営業担当者（経験年数1-5年）
- 主要機能: 提案書構成案作成、競合分析、価格設定支援

■ ペルソナ設定
- 役割: 経験豊富な営業マネージャー
- 口調: 親しみやすく、実践的
- 専門性: BtoB営業、業界動向、価格戦略

■ 制約・ガイドライン
- 競合他社の機密情報は使用禁止
- 価格情報は概算のみ提供
- 法的リスクを伴う提案は行わない
```

#### 2. システムプロンプト設計
```
あなたは10年以上の営業経験を持つシニア営業マネージャーです。

【あなたの役割】
1. 営業提案書の構成・内容アドバイス
2. 顧客ニーズに基づく価値提案作成
3. 競合分析と差別化戦略提案
4. クロージングに向けた戦略立案

【対話スタイル】
- 親しみやすく、実践的なアドバイス
- 具体的な事例を交えた説明
- 段階的なアプローチで課題解決

【制約事項】
- 特定企業の機密情報は推測で言及しない
- 価格は市場相場範囲での提案のみ
- 法的・倫理的問題のある提案は行わない
- 分からないことは素直に「確認が必要」と伝える

【出力形式】
提案時は以下の構成で回答：
1. 現状分析
2. 課題特定
3. 解決策提案
4. 期待効果
5. 次のアクション
```

### 実践演習: 業務特化GPTs開発

#### プロジェクト例: 「法務チェックアシスタント」
```
Phase 1: 要件分析（1-2時間）
- 法務部門のワークフロー分析
- よくある質問・判断パターンの特定
- リスクレベル分類の設計

Phase 2: ナレッジベース構築（2-3時間）
- 社内規程・ガイドライン整理
- 判例・事例データベース作成
- FAQ・テンプレート準備

Phase 3: GPTs実装（1-2時間）
- システムプロンプト設計
- ファイルアップロード設定
- テスト・調整

Phase 4: 運用・改善（継続）
- ユーザーフィードバック収集
- 精度改善・機能追加
- アップデート管理
```

#### 実装例: システムプロンプト詳細
```
あなたは企業法務の専門家として、契約書レビューと法的リスク判定を行います。

【主要機能】
1. 契約書の条項分析
2. 法的リスクの段階的評価
3. 修正提案・代替案提示
4. 関連法規の確認・解釈

【分析フレームワーク】
■ リスク評価（3段階）
- 高リスク: 重大な法的問題、即座に専門家相談推奨
- 中リスク: 要注意事項、条項修正推奨
- 低リスク: 軽微な改善点、経過観察可

■ 回答構成
1. 【概要】契約の性質・目的
2. 【リスク分析】条項別リスク評価
3. 【修正提案】具体的改善案
4. 【注意事項】特に留意すべき点
5. 【次のアクション】推奨する対応手順

【制約事項】
- 最終判断は必ず弁護士等専門家に確認
- 業界特有の慣行は事前確認を推奨
- 海外法規は専門機関への相談を案内
```

---

## 2. API連携とシステム統合

### AI APIの種類と特徴

#### 1. OpenAI API活用
```python
import openai
from datetime import datetime
import json

class CustomAIAssistant:
    def __init__(self, api_key, system_prompt):
        self.client = openai.OpenAI(api_key=api_key)
        self.system_prompt = system_prompt
        self.conversation_history = []
    
    def chat(self, user_message, context=None):
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # コンテキスト情報の追加
        if context:
            context_message = f"【参考情報】\n{json.dumps(context, ensure_ascii=False, indent=2)}"
            messages.append({"role": "system", "content": context_message})
        
        # 会話履歴の追加
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_message})
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )
        
        assistant_message = response.choices[0].message.content
        
        # 履歴更新
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        
        # 履歴管理（直近10件のみ保持）
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return assistant_message
```

#### 2. 業務システムとの統合例
```python
# CRM連携による営業支援AI
class SalesAI:
    def __init__(self, openai_api_key, crm_config):
        self.ai = CustomAIAssistant(openai_api_key, self.get_sales_prompt())
        self.crm = CRMConnector(crm_config)
    
    def analyze_customer(self, customer_id):
        # CRMから顧客データ取得
        customer_data = self.crm.get_customer_data(customer_id)
        deal_history = self.crm.get_deal_history(customer_id)
        interaction_log = self.crm.get_interaction_log(customer_id)
        
        context = {
            "customer_info": customer_data,
            "purchase_history": deal_history,
            "communication_log": interaction_log
        }
        
        analysis_request = """
        この顧客に対する最適なアプローチ戦略を分析してください：
        1. 顧客の特徴・ニーズ分析
        2. 過去の取引パターン分析
        3. 最適な提案タイミング
        4. 推奨アプローチ方法
        5. 想定される課題と対策
        """
        
        return self.ai.chat(analysis_request, context)
    
    def get_sales_prompt(self):
        return """
        あなたは経験豊富な営業戦略コンサルタントです。
        CRMデータを分析し、顧客に最適な営業アプローチを提案します。
        
        分析観点：
        - 購買行動パターン
        - 意思決定プロセス
        - 予算規模・サイクル
        - 競合状況
        - 関係性の深度
        
        提案内容：
        - 具体的なアプローチ手順
        - タイミング戦略
        - 訴求ポイント
        - リスク要因と対策
        - 成功確率向上策
        """
```

### 実践プロジェクト: 統合AIシステム開発

#### プロジェクト例: 「顧客サポート自動化システム」
```python
# 統合顧客サポートAI
class CustomerSupportAI:
    def __init__(self, config):
        self.ai = CustomAIAssistant(config['openai_key'], self.get_support_prompt())
        self.knowledge_base = KnowledgeBaseConnector(config['kb_config'])
        self.ticket_system = TicketSystemConnector(config['ticket_config'])
        self.escalation_rules = config['escalation_rules']
    
    def handle_inquiry(self, inquiry_text, customer_id=None):
        # 1. 関連情報の収集
        relevant_docs = self.knowledge_base.search(inquiry_text)
        customer_context = self.get_customer_context(customer_id) if customer_id else None
        
        # 2. AI分析・回答生成
        context = {
            "knowledge_base": relevant_docs,
            "customer_context": customer_context,
            "escalation_criteria": self.escalation_rules
        }
        
        response = self.ai.chat(f"顧客からの問い合わせ: {inquiry_text}", context)
        
        # 3. エスカレーション判定
        if self.should_escalate(inquiry_text, response):
            return self.create_escalation_ticket(inquiry_text, customer_id, response)
        
        # 4. 自動回答
        return {
            "type": "automated_response",
            "content": response,
            "confidence": self.calculate_confidence(inquiry_text, relevant_docs),
            "follow_up_required": self.needs_follow_up(response)
        }
    
    def should_escalate(self, inquiry, ai_response):
        escalation_keywords = [
            "法的問題", "契約解除", "返金", "不具合", "データ漏洩",
            "緊急", "至急", "クレーム", "苦情"
        ]
        
        # キーワードベース判定
        for keyword in escalation_keywords:
            if keyword in inquiry:
                return True
        
        # AI信頼度ベース判定
        confidence_indicators = ["確認が必要", "専門家への相談", "詳細調査"]
        for indicator in confidence_indicators:
            if indicator in ai_response:
                return True
        
        return False
```

---

## 3. 機械学習基礎と実装

### ビジネス向け機械学習の基礎概念

#### 1. 教師あり学習の業務応用
```python
# 顧客離反予測モデル
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class ChurnPredictionModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_columns = None
        
    def prepare_data(self, customer_data):
        """顧客データの前処理"""
        # 特徴量エンジニアリング
        features = customer_data.copy()
        
        # 数値特徴量
        features['tenure_months'] = features['tenure_months']
        features['monthly_charges'] = features['monthly_charges']
        features['total_charges'] = pd.to_numeric(features['total_charges'], errors='coerce')
        
        # カテゴリ特徴量のエンコーディング
        categorical_features = ['contract_type', 'payment_method', 'internet_service']
        features_encoded = pd.get_dummies(features, columns=categorical_features)
        
        # 派生特徴量
        features_encoded['charges_per_month'] = features_encoded['total_charges'] / features_encoded['tenure_months']
        features_encoded['contract_value'] = features_encoded['monthly_charges'] * features_encoded['tenure_months']
        
        return features_encoded
    
    def train(self, training_data, target_column='churned'):
        """モデル訓練"""
        X = self.prepare_data(training_data.drop(columns=[target_column]))
        y = training_data[target_column]
        
        self.feature_columns = X.columns.tolist()
        
        # 訓練・テスト分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # モデル訓練
        self.model.fit(X_train, y_train)
        
        # 評価
        y_pred = self.model.predict(X_test)
        print("モデル評価結果:")
        print(classification_report(y_test, y_pred))
        
        # 特徴量重要度
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n特徴量重要度 Top 10:")
        print(feature_importance.head(10))
        
        return self.model
    
    def predict_churn_risk(self, customer_data):
        """離反リスク予測"""
        X = self.prepare_data(customer_data)[self.feature_columns]
        
        # 離反確率
        churn_probability = self.model.predict_proba(X)[:, 1]
        
        # リスク分類
        risk_levels = []
        for prob in churn_probability:
            if prob >= 0.8:
                risk_levels.append("高リスク")
            elif prob >= 0.5:
                risk_levels.append("中リスク")
            else:
                risk_levels.append("低リスク")
        
        results = pd.DataFrame({
            'customer_id': customer_data.index,
            'churn_probability': churn_probability,
            'risk_level': risk_levels
        })
        
        return results
    
    def save_model(self, filepath):
        """モデル保存"""
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns
        }, filepath)
    
    def load_model(self, filepath):
        """モデル読み込み"""
        saved_model = joblib.load(filepath)
        self.model = saved_model['model']
        self.feature_columns = saved_model['feature_columns']
```

#### 2. AI予測結果の業務活用
```python
# 予測結果を使った自動アクションシステム
class CustomerRetentionSystem:
    def __init__(self, churn_model, crm_connector, email_system):
        self.churn_model = churn_model
        self.crm = crm_connector
        self.email = email_system
    
    def daily_churn_analysis(self):
        """日次離反リスク分析・対応"""
        # 全顧客データ取得
        customer_data = self.crm.get_all_customers()
        
        # 離反リスク予測
        risk_predictions = self.churn_model.predict_churn_risk(customer_data)
        
        # リスクレベル別対応
        high_risk_customers = risk_predictions[risk_predictions['risk_level'] == '高リスク']
        medium_risk_customers = risk_predictions[risk_predictions['risk_level'] == '中リスク']
        
        # 高リスク顧客への即座対応
        for _, customer in high_risk_customers.iterrows():
            self.handle_high_risk_customer(customer)
        
        # 中リスク顧客への予防的アプローチ
        for _, customer in medium_risk_customers.iterrows():
            self.handle_medium_risk_customer(customer)
        
        # レポート生成
        self.generate_daily_report(risk_predictions)
    
    def handle_high_risk_customer(self, customer):
        """高リスク顧客対応"""
        customer_info = self.crm.get_customer_details(customer['customer_id'])
        
        # 営業担当者への緊急アラート
        alert_message = f"""
        【緊急】顧客離反リスク アラート
        
        顧客ID: {customer['customer_id']}
        顧客名: {customer_info['name']}
        離反確率: {customer['churn_probability']:.1%}
        
        推奨アクション:
        1. 48時間以内の電話でのヒアリング
        2. 特別割引・特典の提案
        3. サービス改善要望の確認
        
        詳細: {customer_info['profile_url']}
        """
        
        self.email.send_urgent_alert(customer_info['account_manager'], alert_message)
        
        # CRMにタスク登録
        self.crm.create_task({
            'customer_id': customer['customer_id'],
            'type': 'churn_prevention',
            'priority': 'urgent',
            'description': '離反リスク高 - 緊急対応要',
            'due_date': datetime.now() + timedelta(days=2)
        })
```

---

## 4. セキュリティとプライバシー

### AIシステムのセキュリティ対策

#### 1. データプライバシー保護
```python
# プライバシー保護機能付きAIシステム
import hashlib
import re
from cryptography.fernet import Fernet

class PrivacyProtectedAI:
    def __init__(self, encryption_key=None):
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
        # 個人情報検出パターン
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}-\d{4}-\d{4}\b|\b\d{11}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b'
        }
    
    def sanitize_input(self, text):
        """個人情報の検出・匿名化"""
        sanitized = text
        detected_pii = []
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                # ハッシュ化による匿名化
                hashed = hashlib.sha256(match.encode()).hexdigest()[:8]
                placeholder = f"[{pii_type.upper()}_{hashed}]"
                sanitized = sanitized.replace(match, placeholder)
                detected_pii.append({'type': pii_type, 'original': match})
        
        return sanitized, detected_pii
    
    def secure_ai_query(self, query_text):
        """セキュアなAIクエリ実行"""
        # 1. 入力の匿名化
        sanitized_query, pii_detected = self.sanitize_input(query_text)
        
        # 2. PII検出時の警告
        if pii_detected:
            print(f"注意: 個人情報が検出されました: {[p['type'] for p in pii_detected]}")
        
        # 3. AIクエリ実行（匿名化済みデータで）
        ai_response = self.query_ai(sanitized_query)
        
        # 4. ログの暗号化保存
        self.log_encrypted_query(sanitized_query, ai_response)
        
        return ai_response, pii_detected
    
    def log_encrypted_query(self, query, response):
        """暗号化ログ保存"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response
        }
        encrypted_log = self.cipher.encrypt(json.dumps(log_data).encode())
        
        # ログファイルへの保存（実装例）
        with open('encrypted_ai_logs.dat', 'ab') as f:
            f.write(encrypted_log + b'\n')
```

#### 2. アクセス制御・認証
```python
# ロールベースアクセス制御
class AISystemAuth:
    def __init__(self):
        self.user_roles = {
            'admin': ['full_access', 'user_management', 'system_config'],
            'manager': ['data_analysis', 'report_generation', 'team_data'],
            'employee': ['basic_query', 'own_data_only']
        }
        
        self.data_access_levels = {
            'public': 0,
            'internal': 1,
            'confidential': 2,
            'restricted': 3
        }
    
    def check_permission(self, user_role, requested_action, data_level):
        """権限チェック"""
        if user_role not in self.user_roles:
            return False
        
        user_permissions = self.user_roles[user_role]
        
        # アクション権限チェック
        if requested_action not in user_permissions and 'full_access' not in user_permissions:
            return False
        
        # データレベル権限チェック
        max_data_level = self.get_max_data_level(user_role)
        if self.data_access_levels[data_level] > max_data_level:
            return False
        
        return True
    
    def get_max_data_level(self, user_role):
        """ロール別最大データアクセスレベル"""
        level_mapping = {
            'admin': 3,
            'manager': 2,
            'employee': 1
        }
        return level_mapping.get(user_role, 0)
```

### AIシステム監査・コンプライアンス

#### 1. AI決定プロセスの透明性確保
```python
class AIDecisionTracker:
    def __init__(self):
        self.decision_log = []
    
    def track_decision(self, input_data, model_output, context):
        """AI決定の追跡"""
        decision_record = {
            'timestamp': datetime.now().isoformat(),
            'input_hash': hashlib.sha256(str(input_data).encode()).hexdigest(),
            'model_version': context.get('model_version'),
            'confidence_score': context.get('confidence'),
            'decision_factors': context.get('feature_importance', {}),
            'output': model_output,
            'reviewer': None,  # 後から人間のレビューを追加
            'approved': None
        }
        
        self.decision_log.append(decision_record)
        return decision_record['timestamp']
    
    def generate_audit_report(self, start_date, end_date):
        """監査レポート生成"""
        relevant_decisions = [
            d for d in self.decision_log
            if start_date <= d['timestamp'] <= end_date
        ]
        
        report = {
            'period': f"{start_date} to {end_date}",
            'total_decisions': len(relevant_decisions),
            'avg_confidence': sum(d.get('confidence_score', 0) for d in relevant_decisions) / len(relevant_decisions),
            'review_rate': len([d for d in relevant_decisions if d['reviewer']]) / len(relevant_decisions),
            'approval_rate': len([d for d in relevant_decisions if d['approved']]) / len(relevant_decisions)
        }
        
        return report
```

---

## まとめ

### 本章で習得したスキル

1. **GPTs開発**: 業務特化型AIアシスタントの設計・実装
2. **API統合**: 既存システムとAIの効果的な連携
3. **機械学習実装**: ビジネス課題解決のための予測モデル構築
4. **セキュリティ対策**: プライバシー保護とアクセス制御の実装

### 実践への活用指針

#### 短期目標（1-3ヶ月）
- 小規模GPTsプロトタイプの開発
- 既存システムとの簡単なAPI連携実装
- 社内データでの予測モデル検証

#### 中期目標（3-6ヶ月）
- 本格的な業務システム統合
- カスタムAIソリューションの運用開始
- セキュリティ・コンプライアンス体制整備

#### 長期目標（6-12ヶ月）
- 組織全体でのAI活用文化醸成
- 独自AIプロダクトの開発・商用化
- AI専門チームの構築・指導

### 注意点・リスク管理

1. **技術的リスク**: モデルの偏見、過学習、データ品質
2. **セキュリティリスク**: データ漏洩、不正アクセス、プライバシー侵害
3. **運用リスク**: システム依存、スキル属人化、変化への適応
4. **倫理・法的リスク**: 説明責任、公平性、規制遵守

---

## 参考資料
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [AI Ethics Guidelines](https://ai-ethics.org/)
- [GDPR AI Compliance Framework](https://gdpr.eu/artificial-intelligence/)