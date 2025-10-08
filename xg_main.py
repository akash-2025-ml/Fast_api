import os
import warnings
import absl.logging

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Suppress protobuf mismatch warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, module="google.protobuf.runtime_version"
)

# Hide absl warnings
absl.logging.set_verbosity(absl.logging.ERROR)


from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import tensorflow as tf
import pickle
import shap

# uvicorn ann_main:app --reload
# Load saved objects

# model = tf.keras.models.load_model("ann_model_23-9-2025_V2.h5")


with open("xg_model_7-10-2025.pkl", "rb") as f:
    xg = pickle.load(f)

svd = joblib.load("truncated_svd_model.pkl")
with open("label_encoder_new_1_06-10-2025.pkl", "rb") as f:
    label_encoders = pickle.load(f)  # Now this is a dict

with open("label_encoder_8-10-2025.pkl", "rb") as f:
    label_encoders_output = pickle.load(f)

with open("explainer.pkl", "rb") as f:
    explainer = pickle.load(f)

########################################################################


categorical_columns = [
    "request_type",
    "spf_result",
    "dkim_result",
    "dmarc_result",
    "tls_version",
    "ssl_validity_status",
    "unique_parent_process_names",
]

################################################################################
feature_names = [
    "sender_known_malicios",
    "sender_domain_reputation_score",
    "sender_spoof_detected",
    "sender_temp_email_likelihood",
    "dmarc_enforced",
    "packer_detected",
    "any_file_hash_malicious",
    "max_metadata_suspicious_score",
    "malicious_attachment_Count",
    "has_executable_attachment",
    "unscannable_attachment_present",
    "total_yara_match_count",
    "total_ioc_count",
    "max_behavioral_sandbox_score",
    "max_amsi_suspicion_score",
    "any_macro_enabled_document",
    "any_vbscript_javascript_detected",
    "any_active_x_objects_detected",
    "any_network_call_on_open",
    "max_exfiltration_behavior_score",
    "any_exploit_pattern_detected",
    "total_embedded_file_count",
    "max_suspicious_string_entropy_score",
    "max_sandbox_execution_time",
    "unique_parent_process_names",
    "return_path_mismatch_with_from",
    "return_path_known_malicious",
    "return_path_reputation_score",
    "reply_path_known_malicious",
    "reply_path_diff_from_sender",
    "reply_path_reputation_Score",
    "smtp_ip_known_malicious",
    "smtp_ip_geo",
    "smtp_ip_asn",
    "smtp_ip_reputation_score",
    "domain_known_malicious",
    "url_Count",
    "dns_morphing_detected",
    "domain_tech_stack_match_score",
    "is_high_risk_role_targeted",
    "sender_name_similarity_to_vip",
    "urgency_keywords_present",
    "request_type",
    "content_spam_score",
    "user_marked_as_spam_before",
    "bulk_message_indicator",
    "unsubscribe_link_present",
    "marketing-keywords_detected",
    "html_text_ratio",
    "image_only_email",
    "spf_result",
    "dkim_result",
    "dmarc_result",
    "reverse_dns_valid",
    "tls_version",
    "total_links_detected",
    "url_shortener_detected",
    "url_redirect_chain_length",
    "final_url_known_malicious",
    "url_decoded_spoof_detected",
    "url_reputation_score",
    "ssl_validity_status",
    "site_visual_similarity_to_known_brand",
    "url_rendering_behavior_score",
    "link_rewritten_through_redirector",
    "token_validation_success",
    "total_components_detected_malicious",
    "Analysis_of_the_qrcode_if_present",
]


##############################################################################################
default_values = {col: 0 for col in feature_names}  # default numeric value

default_values.update(
    {
        "sender_known_malicios": 0,
        "sender_domain_reputation_score": 0.95,
        "sender_spoof_detected": 0,
        "sender_temp_email_likelihood": 0.0,
        "dmarc_enforced": 1,
        "packer_detected": 0,
        "any_file_hash_malicious": 0,
        "max_metadata_suspicious_score": 0.0,
        "malicious_attachment_Count": 0,
        "has_executable_attachment": 0,
        "unscannable_attachment_present": 0,
        "total_yara_match_count": 0,
        "total_ioc_count": 0,
        "max_behavioral_sandbox_score": 0.0,
        "max_amsi_suspicion_score": 0.0,
        "any_macro_enabled_document": 0,
        "any_vbscript_javascript_detected": 0,
        "any_active_x_objects_detected": 0,
        "any_network_call_on_open": 0,
        "max_exfiltration_behavior_score": 0.0,
        "any_exploit_pattern_detected": 0,
        "total_embedded_file_count": 0,
        "max_suspicious_string_entropy_score": 0,
        "max_sandbox_execution_time": 4.377776483e-107,
        "unique_parent_process_names": '[""]',
        "return_path_mismatch_with_from": 0,
        "return_path_known_malicious": 0,
        "return_path_reputation_score": 0.95,
        "reply_path_known_malicious": 0,
        "reply_path_diff_from_sender": 0,
        "reply_path_reputation_Score": 0.95,
        "smtp_ip_known_malicious": 0,
        "smtp_ip_geo": 0.95,
        "smtp_ip_asn": 0.95,
        "smtp_ip_reputation_score": 0.95,
        "domain_known_malicious": 0,
        "url_Count": 0,
        "dns_morphing_detected": 0,
        "domain_tech_stack_match_score": 1.0,
        "is_high_risk_role_targeted": 0,
        "sender_name_similarity_to_vip": 0.0000000,
        "urgency_keywords_present": 0,
        "request_type": "none",
        "content_spam_score": 0.0,
        "user_marked_as_spam_before": 0,
        "bulk_message_indicator": 0,
        "unsubscribe_link_present": 0,
        "marketing-keywords_detected": 0,
        "html_text_ratio": 0.0,
        "image_only_email": 0,
        "spf_result": "pass",
        "dkim_result": "pass",
        "dmarc_result": "pass",
        "reverse_dns_valid": 1,
        "tls_version": "TLS 1.0",
        "total_links_detected": 0,
        "url_shortener_detected": 0,
        "url_redirect_chain_length": 0,
        "final_url_known_malicious": 0,
        "url_decoded_spoof_detected": 0,
        "url_reputation_score": 0.0,
        "ssl_validity_status": "valid",
        "site_visual_similarity_to_known_brand": 0.6138188595,
        "url_rendering_behavior_score": 0.0000430629,
        "link_rewritten_through_redirector": 0,
        "token_validation_success": 1,
        "total_components_detected_malicious": 0,
        "Analysis_of_the_qrcode_if_present": 0,
    }
)

################################################################################################
app = FastAPI()
l = []


# Input schema
class InputData(BaseModel):
    data: dict  # {"feature_name": value, ...}


@app.get("/")
def home():
    return {"message": "Welcome to the ML API! Send POST to /predict with JSON data."}


@app.post("/predict")
def predict(input_data: InputData):
    row = []
    for col in feature_names:
        value = input_data.data.get(col, default_values.get(col))

        if col in categorical_columns:
            encoder = label_encoders[col]
            try:
                value = encoder.transform([value])[0]
            except ValueError:
                value = -1

        row.append(value)

    X = np.array(row).reshape(1, -1)

    # Apply SVD
    X_svd = svd.transform(X)
    # xg_pred = xg.predict(X_svd)
    # X_svd = X_svd.astype("float32")
    # # Predict
    # probs = model.predict(X_svd)
    probs = xg.predict(X_svd)
    pred_class = np.argmax(probs)
    original_labels = label_encoders_output.inverse_transform([pred_class])

    # shap

    # explainer = shap.Explainer(model, X_svd)
    # shap_values = explainer(X_svd)  # shape = (samples, features, classes)
    # a = int(pred_class)
    # abs_vals = np.abs(shap_values[:, :, a].values[0])
    # sorted_idx = np.argsort(abs_vals)[::-1]
    # for i in sorted_idx[:10]:
    #     # print(f" - {feature_names[i]} === ")
    #     l.append(feature_names[i])

    # return {"predicted_class": original_labels, "predicted_class by XG": str(probs)}
    return {"predicted_class": str(original_labels[0])}
    # return {"predicted_class": str(value)}
