from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import tensorflow as tf
import pickle
import shap

# Load saved objects

model = tf.keras.models.load_model("ann_model.h5")
svd = joblib.load("svd.pkl")

with open("label_encoder_new.pkl", "rb") as f:
    label_encoders = pickle.load(f)  # Now this is a dict

with open("label_encoder.pkl", "rb") as f:
    label_encoders_output = pickle.load(f)

with open("explainer.pkl", "rb") as f:
    explainer = pickle.load(f)

########################################################################
feature_columns = [
    "File?",
    "sender_known_malicios",
    "sender_domain_reputation_score(float)",
    "sender_spoof_detected(bool)",
    "sender_temp_email_likelihood(float)",
    "dmarc_enforced(bool)",
    "packer_detected(bool)",
    "any_file_hash_malicious(bool)",
    "max_metadata_suspicious_score(float)",
    "malicious_attachment_Count(int)",
    "has_executable_attachment(bool)",
    "unscannable_attachment_present(bool)",
    "Authentication & Identity Assurance",
    "total_yara_match_count(int)",
    "total_ioc_count(int)",
    "max_behavioral_sandbox_score(float)",
    "max_amsi_suspicion_score(float)",
    "any_macro_enabled_document(bool)",
    "any_vbscript_javascript_detected(bool)",
    "any_active_x_objects_detected(bool)",
    "any_network_call_on_open(bool)",
    "max_exfiltration_behavior_score(float)",
    "any_exploit_pattern_detected(bool)",
    "total_embedded_file_count(int)",
    "max_suspicious_string_entropy_score(float)",
    "max_sandbox_execution_time(float)",
    "return_path_mismatch_with_from(bool)",
    "return_path_known_malicious(bool)",
    "return_path_reputation_score(float)",
    "reply_path_known_malicious(bool)",
    "reply_path_diff_from_sender(bool)",
    "reply_path_reputation_Score(float)",
    "smtp_ip_known_malicious(bool)",
    "smtp_ip_geo(Float)",
    "smtp_ip_asn(categorical)",
    "smtp_ip_reputation_score(float)",
    "domain_known_malicious(bool)",
    "url_Count(int)",
    "dna_morphing_detected(bool)",
    "domain_tech_stack_match_score(float)",
    "is_high_risk_role_targeted(bool)",
    "sender_name_similarity_to_vip(float)",
    "urgency_keywords_present(bool)",
    "request_type(categorical)",
    "content_spam_score(float)",
    "user_marked_as_spam_before(bool)",
    "bulk_message_indicator(bool)",
    "unsubscribe_link_present(bool)",
    "marketing-keywords_detected(bool)",
    "html_text_ratio(float)",
    "image_only_email(bool)",
    "spf_result(enum)",
    "dkim_result(enum)",
    "dmarc_result(enum)",
    "reverse_dns_valid(bool)",
    "tls_version(ctegorical)",
    "total_links_detected(int)",
    "url_shortener_detected(bool)",
    "url_redirect_chain_length(int)",
    "final_url_known_malicious(bool)",
    "url_decoded_spoof_detected(bool)",
    "url_reputation_score(float)",
    "ssl_validity_status(enum)",
    "site_visual_similarity_to_known_brand(float)",
    "url_rendering_behavior_score(float)",
    "link_rewritten_through_redirector(bool)",
    "token_validation_success(bool)",
    "Analysis_of_the_qrcode_if_present(verdict-malicious or non-malicious)",
]

categorical_columns = [
    "request_type(categorical)",
    "spf_result(enum)",
    "dkim_result(enum)",
    "dmarc_result(enum)",
    "tls_version(ctegorical)",
    "ssl_validity_status(enum)",
]

################################################################################
feature_names = [
    "File?",
    "sender_known_malicios",
    "sender_domain_reputation_score(float)",
    "sender_spoof_detected(bool)",
    "sender_temp_email_likelihood(float)",
    "dmarc_enforced(bool)",
    "packer_detected(bool)",
    "any_file_hash_malicious(bool)",
    "max_metadata_suspicious_score(float)",
    "malicious_attachment_Count(int)",
    "has_executable_attachment(bool)",
    "unscannable_attachment_present(bool)",
    "Authentication & Identity Assurance",
    "total_yara_match_count(int)",
    "total_ioc_count(int)",
    "max_behavioral_sandbox_score(float)",
    "max_amsi_suspicion_score(float)",
    "any_macro_enabled_document(bool)",
    "any_vbscript_javascript_detected(bool)",
    "any_active_x_objects_detected(bool)",
    "any_network_call_on_open(bool)",
    "max_exfiltration_behavior_score(float)",
    "any_exploit_pattern_detected(bool)",
    "total_embedded_file_count(int)",
    "max_suspicious_string_entropy_score(float)",
    "max_sandbox_execution_time(float)",
    "return_path_mismatch_with_from(bool)",
    "return_path_known_malicious(bool)",
    "return_path_reputation_score(float)",
    "reply_path_known_malicious(bool)",
    "reply_path_diff_from_sender(bool)",
    "reply_path_reputation_Score(float)",
    "smtp_ip_known_malicious(bool)",
    "smtp_ip_geo(Float)",
    "smtp_ip_asn(categorical)",
    "smtp_ip_reputation_score(float)",
    "domain_known_malicious(bool)",
    "url_Count(int)",
    "dna_morphing_detected(bool)",
    "domain_tech_stack_match_score(float)",
    "is_high_risk_role_targeted(bool)",
    "sender_name_similarity_to_vip(float)",
    "urgency_keywords_present(bool)",
    "request_type(categorical)",
    "content_spam_score(float)",
    "user_marked_as_spam_before(bool)",
    "bulk_message_indicator(bool)",
    "unsubscribe_link_present(bool)",
    "marketing-keywords_detected(bool)",
    "html_text_ratio(float)",
    "image_only_email(bool)",
    "spf_result(enum)",
    "dkim_result(enum)",
    "dmarc_result(enum)",
    "reverse_dns_valid(bool)",
    "tls_version(ctegorical)",
    "total_links_detected(int)",
    "url_shortener_detected(bool)",
    "url_redirect_chain_length(int)",
    "final_url_known_malicious(bool)",
    "url_decoded_spoof_detected(bool)",
    "url_reputation_score(float)",
    "ssl_validity_status(enum)",
    "site_visual_similarity_to_known_brand(float)",
    "url_rendering_behavior_score(float)",
    "link_rewritten_through_redirector(bool)",
    "token_validation_success(bool)",
    "Analysis_of_the_qrcode_if_present(verdict-malicious or non-malicious)",
]


##############################################################################################
default_values = {col: 0 for col in feature_columns}  # default numeric value

default_values.update(
    {
        "File?": 1.0,
        "sender_known_malicios": 0,
        "sender_domain_reputation_score(float)": 0.0264308467,
        "sender_spoof_detected(bool)": 0,
        "sender_temp_email_likelihood(float)": 0.1315211125,
        "dmarc_enforced(bool)": 0,
        "packer_detected(bool)": 0,
        "any_file_hash_malicious(bool)": 0,
        "max_metadata_suspicious_score(float)": 0.0001340762,
        "malicious_attachment_Count(int)": 0,
        "has_executable_attachment(bool)": 0,
        "unscannable_attachment_present(bool)": 0,
        "Authentication & Identity Assurance": 0,
        "total_yara_match_count(int)": 0,
        "total_ioc_count(int)": 0,
        "max_behavioral_sandbox_score(float)": 0.6053533113,
        "max_amsi_suspicion_score(float)": 0.0000037708,
        "any_macro_enabled_document(bool)": 0,
        "any_vbscript_javascript_detected(bool)": 0,
        "any_active_x_objects_detected(bool)": 0,
        "any_network_call_on_open(bool)": 0,
        "max_exfiltration_behavior_score(float)": 0.6370384566,
        "any_exploit_pattern_detected(bool)": 0,
        "total_embedded_file_count(int)": 0,
        "max_suspicious_string_entropy_score(float)": 7,
        "max_sandbox_execution_time(float)": 4.377776483e-107,
        "return_path_mismatch_with_from(bool)": 0,
        "return_path_known_malicious(bool)": 0,
        "return_path_reputation_score(float)": 0.0000000088,
        "reply_path_known_malicious(bool)": 0,
        "reply_path_diff_from_sender(bool)": 0,
        "reply_path_reputation_Score(float)": 0.0000640285,
        "smtp_ip_known_malicious(bool)": 0,
        "smtp_ip_geo(Float)": 0.0014695575,
        "smtp_ip_asn(categorical)": 0.000032619,
        "smtp_ip_reputation_score(float)": 0.0000895979,
        "domain_known_malicious(bool)": 0,
        "url_Count(int)": 0,
        "dna_morphing_detected(bool)": 0,
        "domain_tech_stack_match_score(float)": 1.0,
        "is_high_risk_role_targeted(bool)": 0,
        "sender_name_similarity_to_vip(float)": 0.0000000001,
        "urgency_keywords_present(bool)": 0,
        "request_type(categorical)": "none",
        "content_spam_score(float)": 0.0004012214,
        "user_marked_as_spam_before(bool)": 0,
        "bulk_message_indicator(bool)": 0,
        "unsubscribe_link_present(bool)": 0,
        "marketing-keywords_detected(bool)": 0,
        "html_text_ratio(float)": 0.0,
        "image_only_email(bool)": 0,
        "spf_result(enum)": "fail",
        "dkim_result(enum)": "pass",
        "dmarc_result(enum)": "none",
        "reverse_dns_valid(bool)": 1,
        "tls_version(ctegorical)": "TLS 1.0",
        "total_links_detected(int)": 0,
        "url_shortener_detected(bool)": 0,
        "url_redirect_chain_length(int)": 0,
        "final_url_known_malicious(bool)": 0,
        "url_decoded_spoof_detected(bool)": 0,
        "url_reputation_score(float)": 0.0,
        "ssl_validity_status(enum)": "valid",
        "site_visual_similarity_to_known_brand(float)": 0.6138188595,
        "url_rendering_behavior_score(float)": 0.0000430629,
        "link_rewritten_through_redirector(bool)": 0,
        "token_validation_success(bool)": 1,
        "Analysis_of_the_qrcode_if_present(verdict-malicious or non-malicious)": 0,
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
    for col in feature_columns:
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
    X_svd = X_svd.astype("float32")
    # Predict
    probs = model.predict(X_svd)
    pred_class = np.argmax(probs, axis=1)
    original_labels = label_encoders_output.inverse_transform(pred_class)

    # shap

    explainer = shap.Explainer(model, X_svd)
    shap_values = explainer(X_svd)  # shape = (samples, features, classes)
    a = int(pred_class)
    abs_vals = np.abs(shap_values[:, :, a].values[0])
    sorted_idx = np.argsort(abs_vals)[::-1]
    for i in sorted_idx[:10]:
        # print(f" - {feature_names[i]} === ")
        l.append(feature_names[i])

    return {
        "predicted_class": original_labels[0],
        "Top features influencing prediction": str(l),
    }

    # return {"predicted_class": pred_class, "prediction_probability": pred_prob}
