"""
RetailPulse - Customer Segmentation Module
Combines RFM scoring with K-Means and DBSCAN clustering.
Produces 6-8 business-interpretable customer segments and persists model artifacts.
"""
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import config
from src.data.load_data import load_clean_data
from src.features.rfm import calculate_rfm

logger = logging.getLogger(__name__)


class CustomerSegmenter:
    """
    Production segmentation pipeline using Scaled RFM features and K-Means.
    """
    def __init__(self, n_clusters: int = 6, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.dbscan = DBSCAN(eps=0.8, min_samples=10)
        self.feature_cols = ["Recency_Log", "Frequency_Log", "Monetary_Log"]
        self.is_fitted = False

    def fit_predict(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        df = rfm_df.copy()
        # Log-transform to handle right skewness in retail spend
        df["Recency_Log"] = np.log1p(np.maximum(df["Recency"], 0))
        df["Frequency_Log"] = np.log1p(np.maximum(df["Frequency"], 0))
        df["Monetary_Log"] = np.log1p(np.maximum(df["Monetary"], 0))

        X = df[self.feature_cols].values
        X_scaled = self.scaler.fit_transform(X)

        # K-Means clustering
        df["Cluster"] = self.kmeans.fit_predict(X_scaled)
        self.is_fitted = True

        # DBSCAN outlier detection (-1 indicates anomaly / noise)
        df["DBSCAN_Outlier"] = self.dbscan.fit_predict(X_scaled) == -1

        # Calculate Silhouette Score
        sil_score = silhouette_score(X_scaled, df["Cluster"])
        logger.info(f"K-Means fitted with {self.n_clusters} clusters. Silhouette Score: {sil_score:.3f}")

        # Map clusters to descriptive persona names based on centroid rankings
        cluster_means = df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
        # Rank by overall value score
        cluster_means["ValueRank"] = (
            -cluster_means["Recency"].rank() +
            cluster_means["Frequency"].rank() * 2 +
            cluster_means["Monetary"].rank() * 2
        )
        sorted_clusters = cluster_means["ValueRank"].sort_values(ascending=False).index.tolist()

        persona_names = [
            "Champions (VIP)",
            "Loyal High-Spenders",
            "Potential Loyalists",
            "Promising New Customers",
            "At Risk / Drifted",
            "Hibernating / Inactive",
            "Lost Customers",
            "Occasional Bargain Shoppers"
        ]

        self.cluster_to_persona = {cluster: persona_names[i % len(persona_names)] for i, cluster in enumerate(sorted_clusters)}
        df["Segment_Name"] = df["Cluster"].map(self.cluster_to_persona)

        return df

    def predict(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Inference on new customer records without refitting centroids.
        """
        df = rfm_df.copy()
        df["Recency_Log"] = np.log1p(np.maximum(df["Recency"], 0))
        df["Frequency_Log"] = np.log1p(np.maximum(df["Frequency"], 0))
        df["Monetary_Log"] = np.log1p(np.maximum(df["Monetary"], 0))

        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        df["Cluster"] = self.kmeans.predict(X_scaled)
        if hasattr(self, "cluster_to_persona") and self.cluster_to_persona:
            df["Segment_Name"] = df["Cluster"].map(self.cluster_to_persona)
        else:
            df["Segment_Name"] = "Cluster " + df["Cluster"].astype(str)
        return df

    def save(self, filepath=config.SEGMENTATION_MODEL_FILE):
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        state = {
            "kmeans": self.kmeans,
            "scaler": self.scaler,
            "dbscan": self.dbscan,
            "feature_cols": self.feature_cols,
            "cluster_to_persona": getattr(self, "cluster_to_persona", {}),
            "n_clusters": self.n_clusters
        }
        joblib.dump(state, filepath)
        logger.info(f"CustomerSegmenter saved to {filepath}")

    @classmethod
    def load(cls, filepath=config.SEGMENTATION_MODEL_FILE):
        data = joblib.load(filepath)
        if isinstance(data, dict):
            obj = cls(n_clusters=data.get("n_clusters", 6))
            obj.kmeans = data["kmeans"]
            obj.scaler = data["scaler"]
            obj.dbscan = data["dbscan"]
            obj.feature_cols = data["feature_cols"]
            obj.cluster_to_persona = data.get("cluster_to_persona", {})
            obj.is_fitted = True
            return obj
        return data


def train_segmentation_pipeline() -> pd.DataFrame:
    """
    Executes full segmentation workflow: loads data, computes RFM, fits clustering, saves output.
    """
    df = load_clean_data()
    rfm = calculate_rfm(df)

    segmenter = CustomerSegmenter(n_clusters=6, random_state=42)
    segmented_rfm = segmenter.fit_predict(rfm)
    segmenter.save()

    # Save to CSV and Excel
    segmented_rfm.to_csv(config.CUSTOMER_SEGMENTS_CSV, index=False)
    try:
        segmented_rfm.to_excel(config.CUSTOMER_SEGMENTS_XLSX, index=False)
    except Exception as e:
        logger.warning(f"Could not write Excel: {e}")

    logger.info(f"Customer segmentation saved to {config.CUSTOMER_SEGMENTS_CSV}")
    return segmented_rfm


if __name__ == "__main__":
    res = train_segmentation_pipeline()
    print("Clusters distribution:")
    print(res["Segment_Name"].value_counts())
