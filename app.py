import streamlit as st
from inference import process_single

st.set_page_config(page_title="AI Email Agent", layout="centered")

st.title("AI Email Agent")
st.write("Smart email analysis with task extraction & action suggestions")


option = st.radio("Choose Input Type:", ["Use Sample Emails", "Enter Custom Email"])

sample_emails = [
    "Your order has been delayed due to weather conditions.",
    "Congratulations! You have won a prize.",
    "Meeting scheduled at 3 PM tomorrow.",
    "Your bank account needs verification urgently.",
    "Reminder: Submit your assignment before deadline."
]

email_input = ""

if option == "Use Sample Emails":
    email_input = st.selectbox("Select an Email", sample_emails)
else:
    email_input = st.text_area("Enter Email Content")


if st.button("Process Email"):
    if email_input:
        try:
            result = process_single(email_input)

            st.subheader("AI Analysis")

            
            category = result.get("category", "N/A")
            if category == "spam":
                st.error(f"Category: {category.upper()}")
            elif category == "important":
                st.success(f"Category: {category.upper()}")
            else:
                st.info(f"Category: {category.upper()}")

            
            st.write(f"Action: {result.get('action', 'N/A')}")
            st.write(f"Urgency: {result.get('urgency', 'N/A')}")
            st.write(f"Risk: {result.get('risk', 'N/A')}")

            
            st.write("Tasks:")
            tasks = result.get("tasks", [])
            if tasks:
                for task in tasks:
                    st.write(f"- {task}")
            else:
                st.write("No tasks detected")

            
            st.write(f"Deadline: {result.get('deadline', 'N/A')}")

            
            st.write("Reason:")
            st.write(result.get("reason", "N/A"))

            st.subheader("Smart Insight")
            if result.get("risk") == "phishing":
                st.warning("This email may be a phishing attempt. Avoid clicking links.")
            elif result.get("urgency") == "high":
                st.info("This email needs immediate attention.")
            else:
                st.success("This email looks safe.")

            if "reply" in result:
                st.subheader("Suggested Reply")
                st.text_area("", result["reply"], height=120)

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter/select an email.")
