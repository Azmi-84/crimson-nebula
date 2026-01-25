import pickle
import numpy as np
import pandas as pd
import gradio as gr

# Load the pre-trained model
with open('src/models/crimson_nebula.pkl', 'rb') as f:
    model = pickle.load(f)
    

# Define the prediction function
def predict_crimson_nebula(*inputs):
    input_data = dict(zip(feature_names, inputs))
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return prediction[0]


# Define features name
feature_names = [
  "age",
  "gender",
  "country",
  "urban_rural",
  "income_level",
  "employment_status",
  "education_level",
  "relationship_status",
  "has_children",
  "exercise_hours_per_week",
  "sleep_hours_per_night",
  "diet_quality",
  "smoking",
  "alcohol_frequency",
  "perceived_stress_score",
  "body_mass_index",
  "blood_pressure_systolic",
  "blood_pressure_diastolic",
  "daily_steps_count",
  "weekly_work_hours",
  "hobbies_count",
  "social_events_per_month",
  "books_read_per_year",
  "volunteer_hours_per_month",
  "travel_frequency_per_year",
  "daily_active_minutes_instagram",
  "sessions_per_day",
  "posts_created_per_week",
  "reels_watched_per_day",
  "stories_viewed_per_day",
  "likes_given_per_day",
  "comments_written_per_day",
  "dms_sent_per_week",
  "dms_received_per_week",
  "ads_viewed_per_day",
  "ads_clicked_per_day",
  "time_on_feed_per_day",
  "time_on_explore_per_day",
  "time_on_messages_per_day",
  "time_on_reels_per_day",
  "followers_count",
  "following_count",
  "uses_premium_features",
  "notification_response_rate",
  "account_creation_year",
  "average_session_length_minutes",
  "content_type_preference",
  "preferred_content_theme",
  "privacy_setting_level",
  "two_factor_auth_enabled",
  "biometric_login_used",
  "linked_accounts_count",
  "subscription_status",
  "user_engagement_score"
]

# Define the input and output components
input_components = [
    gr.Slider(10, 70, step=1, label="Age"),
    gr.Dropdown(["Male", "Female", "Non-binary", "Prefer not to say"], label="Gender"),
    gr.Dropdown(["United States", "India", "Brazil", "Other", "United Kingdom", "Canada", "Australia", "South Korea", "Germany", "Japan"], label="Country"),
    gr.Dropdown(["Urban", "Suburban", "Rural"], label="Urban/Rural"),
    gr.Dropdown(["Low", "Lower-middle", "Middle", "Upper-middle", "High"], label="Income Level"),
    gr.Dropdown(["Full-time employed", "Student", "Freelancer", "Unemployed", "Part-time", "Retired"], label="Employment Status"),
    gr.Dropdown(["Bachelor's", "High School", "Some College", "Master's", "Other", "PhD"], label="Education Level"),
    gr.Dropdown(["Single", "Married", "In a relationship", "Divorced", "Widowed"], label="Relationship Status"),
    gr.Dropdown(["False", "True"], label="Has Children"),
    gr.Slider(0, 20, step=1, label="Exercise Hours per Week"),
    gr.Slider(0, 12, step=1, label="Sleep Hours per Night"),
    gr.Dropdown(["Average", "Good", "Poor", "Very Poor", "Excellent"], label="Diet Quality"),
    gr.Dropdown(["Yes", "No", "Former"], label="Smoking"),
    gr.Dropdown(["Rarely", "Never", "Weekly", "Several times a week", "Daily"], label="Alcohol Frequency"),
    gr.Slider(0, 40, step=1, label="Perceived Stress Score"),
    gr.Slider(10, 40, step=1, label="Body Mass Index"),
    gr.Slider(90, 180, step=1, label="Blood Pressure Systolic"),
    gr.Slider(60, 120, step=1, label="Blood Pressure Diastolic"),
    gr.Slider(0, 30000, step=100, label="Daily Steps Count"),
    gr.Slider(0, 100, step=1, label="Weekly Work Hours"),
    gr.Slider(0, 20, step=1, label="Hobbies Count"),
    gr.Slider(0, 30, step=1, label="Social Events per Month"),
    gr.Slider(0, 50, step=1, label="Books Read per Year"),
    gr.Slider(0, 100, step=1, label="Volunteer Hours per Month"),
    gr.Slider(0, 20, step=1, label="Travel Frequency per Year"),
    gr.Slider(0, 300, step=10, label="Daily Active Minutes on Instagram"),
    gr.Slider(0, 50, step=1, label="Sessions per Day"),
    gr.Slider(0, 50, step=1, label="Posts Created per Week"),
    gr.Slider(0, 300, step=10, label="Reels Watched per Day"),
    gr.Slider(0, 500, step=10, label="Stories Viewed per Day"),
    gr.Slider(0, 1000, step=10, label="Likes Given per Day"),
    gr.Slider(0, 100, step=1, label="Comments Written per Day"),
    gr.Slider(0, 200, step=1, label="DMs Sent per Week"),
    gr.Slider(0, 200, step=1, label="DMs Received per Week"),
    gr.Slider(0, 500, step=10, label="Ads Viewed per Day"),
    gr.Slider(0, 100, step=1, label="Ads Clicked per Day"),
    gr.Slider(0, 180, step=5, label="Time on Feed per Day (minutes)"),
    gr.Slider(0, 120, step=5, label="Time on Explore per Day (minutes)"),
    gr.Slider(0, 60, step=5, label="Time on Messages per Day (minutes)"),
    gr.Slider(0, 180, step=5, label="Time on Reels per Day (minutes)"),
    gr.Slider(0, 100000, step=1000, label="Followers Count"),
    gr.Slider(0, 5000, step=100, label="Following Count"),
    gr.Dropdown(["False", "True"], label="Uses Premium Features"),
    gr.Slider(0, 100, step=1, label="Notification Response Rate (%)"),
    gr.Slider(2005, 2024, step=1, label="Account Creation Year"),
    gr.Slider(1, 180, step=1, label="Average Session Length (minutes)"),
    gr.Dropdown(["Reels", "Stories", "Live", "Mixed", "Photos"], label="Content Type Preference"),
    gr.Dropdown(["Fashion", "Travel", "Food", "Fitness", "Art", "Music", "Tech", "Other"], label="Preferred Content Theme"),
    gr.Dropdown(["Public", "Private", "Friends only"], label="Privacy Setting Level"),
    gr.Dropdown(["False", "True"], label="Two-Factor Authentication Enabled"),
    gr.Dropdown(["False", "True"], label="Biometric Login Used"),
    gr.Slider(0, 10, step=1, label="Linked Accounts Count"),
    gr.Dropdown(["Free", "Premium", "Business"], label="Subscription Status"),
    gr.Slider(0, 100, step=1, label="User Engagement Score")
]

output_component = gr.Label(label="Crimson Nebula Prediction")


# Interface
app = gr.Interface(
    fn=predict_crimson_nebula,
    inputs=input_components,
    outputs=output_component,
    title="Crimson Nebula",
    description="Happiness Prediction Model"
)


# Launch the app
if __name__ == "__main__":
    app.launch()