import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import wikipedia
import os
import tiktoken

# --- Constants & Environment ---
MODEL_NAME = "llama-3.1-8b-instant"
VECTOR_DB_PATH = "./faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_REQUEST_TOKENS = 4000

# ------------------- Utility Functions ---
def get_safe_prompt(prompt_template, prompt_variables):
    full_prompt = prompt_template.format(**prompt_variables)
    encoded_prompt = encoding.encode(full_prompt)
    if len(encoded_prompt) > MAX_REQUEST_TOKENS:
        truncated_prompt = encoding.decode(encoded_prompt[:MAX_REQUEST_TOKENS])
        return truncated_prompt
    return full_prompt

@st.cache_resource
def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    if os.path.exists(VECTOR_DB_PATH):
        db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        st.info("Loaded FAISS index from disk.")
    else:
        with open("./cuisine_ingredients_ranked.txt", "r", encoding="utf-8") as f:
            docs = f.readlines()
        docs = [d.strip() for d in docs if d.strip()]
        db = FAISS.from_texts(docs, embeddings)
        db.save_local(VECTOR_DB_PATH)
        st.info("Created and saved new FAISS index.")
    
    return db.as_retriever(search_kwargs={"k": 2})

def reset_session_state():
    """Resets all session state variables to start over."""
    st.session_state["stage"] = "initial"
    st.session_state["names"] = []
    st.session_state["selected_name"] = None
    st.session_state["taglines"] = None
    st.session_state["selected_tagline"] = None
    st.session_state["cuisine"] = None
    st.session_state["vibe"] = None
    st.session_state["show_dishes"] = False
    st.session_state["show_ingredients"] = False
    st.session_state["show_restaurants"] = False
    st.session_state["generated_dishes"] = None
    st.session_state["generated_ingredients"] = None
    st.session_state["generated_restaurants"] = None
    if "chat_history" in st.session_state:
        del st.session_state["chat_history"]

# --- Function to Display Final Content ---
def display_final_content():
    """Displays the selected brand and details from cached variables."""
    if st.session_state["selected_name"]:
        st.markdown(f"<h1 style='text-align: center;'>{st.session_state['selected_name']}</h1>", unsafe_allow_html=True)
        st.write("---")

    if st.session_state["selected_tagline"]:
        tagline_only = st.session_state["selected_tagline"].split(' - ')[0].strip()
        st.markdown(f"<h3 style='text-align: center;'>{tagline_only}</h3>", unsafe_allow_html=True)
    
    if st.session_state["show_dishes"] or st.session_state["show_ingredients"] or st.session_state["show_restaurants"]:
        st.subheader("Info")
        st.markdown("---")
    
    if st.session_state["show_dishes"] and st.session_state["generated_dishes"]:
        st.subheader("🍲 Suggested Dishes")
        st.write(st.session_state["generated_dishes"].strip())

    if st.session_state["show_ingredients"] and st.session_state["generated_ingredients"]:
        st.subheader("🥕 Prominent Ingredients")
        st.write(st.session_state["generated_ingredients"].strip())

    if st.session_state["show_restaurants"] and st.session_state["generated_restaurants"]:
        st.subheader("🌍 Top Restaurants")
        st.write(st.session_state["generated_restaurants"].strip())

# ------------------- Prompt Templates -------------------
names_prompt_template = "Suggest 5 creative and unique restaurant names for a {cuisine} restaurant with a {vibe} vibe. For each name, provide a one-line description. Only provide the list, formatted as 'Name - Description', with each on a new line. Do not include any extra text, introductions, or conclusions."
tagline_prompt_template = "Write 5 catchy taglines for a restaurant named '{restaurant_name}' which serves {cuisine} cuisine. Only provide the taglines, one per line. Do not include any extra text, introductions, or conclusions."
dish_prompt_template = "Suggest 3 unique dishes for a {cuisine} restaurant."
restaurant_prompt_template = "Give me the names of the top 3 most famous restaurants in the world for {cuisine} cuisine."
rag_prompt_template = "Based on the following documents, list 5 prominent ingredients for {cuisine} cuisine without mentioning any frequencies or rankings. Just give the ingredients, one per line:\n\n{documents}"
wikipedia_prompt_template = "From this text about {cuisine} cuisine, list 5 key ingredients. For each, give a one-line reason for its use. Format as 'Ingredient - Reason':\n\n{summary}"

# ------------------- Setup Groq & Token Counter -------------------
if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY not found in .streamlit/secrets.toml")
    st.stop()
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(model=MODEL_NAME, temperature=0.7)
encoding = tiktoken.get_encoding("cl100k_base")

# ------------------- Streamlit UI -------------------
st.title("🍽️ AI Restaurant Brand Generator")

# Initialize session state for multi-stage flow
if "stage" not in st.session_state:
    reset_session_state()

# --- Stage 1: Initial Input ---
if st.session_state["stage"] == "initial":
    st.session_state["cuisine"] = st.text_input("Enter a cuisine:", key="cuisine_input")
    st.session_state["vibe"] = st.text_input("Enter a vibe (e.g., modern, traditional, cozy):", key="vibe_input")
    
    if st.button("Generate Restaurant Names") and st.session_state["cuisine"]:
        with st.spinner("Generating creative names..."):
            try:
                prompt_str = get_safe_prompt(names_prompt_template, {"cuisine": st.session_state["cuisine"], "vibe": st.session_state["vibe"]})
                raw_names = llm.predict(prompt_str)
                st.session_state["names"] = [
                    name.strip() for name in raw_names.split('\n') if '-' in name
                ]
                st.session_state["stage"] = "names_generated"
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- Stage 2: Names Generated ---
elif st.session_state["stage"] == "names_generated":
    st.subheader("Here are 5 name suggestions:")
    
    selected_name_full = st.radio("Please select your preferred name:", st.session_state["names"])
    
    if st.button("Select This Name"):
        st.session_state["selected_name"] = selected_name_full.split(' - ')[0].strip()
        
        with st.spinner("Generating taglines..."):
            try:
                tagline_prompt_str = get_safe_prompt(tagline_prompt_template, {"restaurant_name": st.session_state["selected_name"], "cuisine": st.session_state["cuisine"]})
                raw_taglines = llm.predict(tagline_prompt_str)
                st.session_state["taglines"] = [tagline.strip() for tagline in raw_taglines.split('\n') if tagline.strip()]
                st.session_state["stage"] = "taglines_generated"
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- Stage 3: Taglines Generated ---
elif st.session_state["stage"] == "taglines_generated":
    st.markdown(f"<h1 style='text-align: center;'>{st.session_state.get('selected_name')}</h1>", unsafe_allow_html=True)
    
    selected_tagline = st.radio("Please select your favorite tagline:", st.session_state["taglines"])
    
    if st.button("Confirm Tagline"):
        st.session_state["selected_tagline"] = selected_tagline
        st.session_state["stage"] = "details_selection"
        st.rerun()

# --- Stage 4: Details Selection ---
elif st.session_state["stage"] == "details_selection":
    st.markdown(f"<h1 style='text-align: center;'>{st.session_state.get('selected_name')}</h1>", unsafe_allow_html=True)
    st.write("---")
    tagline_only = st.session_state["selected_tagline"].split(' - ')[0].strip()
    st.markdown(f"<h3 style='text-align: center;'>{tagline_only}</h3>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Would you like more details?")
    
    st.session_state["show_dishes"] = st.checkbox("Show Suggested Dishes")
    st.session_state["show_ingredients"] = st.checkbox("Show Prominent Ingredients")
    st.session_state["show_restaurants"] = st.checkbox("Show Top Restaurants")

    if st.button("Get Details"):
        with st.spinner("Generating all selected details..."):
            if st.session_state["show_dishes"]:
                try:
                    dish_prompt_str = get_safe_prompt(dish_prompt_template, {"cuisine": st.session_state["cuisine"]})
                    st.session_state["generated_dishes"] = llm.predict(dish_prompt_str)
                except Exception as e:
                    st.session_state["generated_dishes"] = f"Error getting dishes: {e}"

            if st.session_state["show_ingredients"]:
                try:
                    retriever = get_retriever()
                    retrieved_docs = retriever.get_relevant_documents(f"ingredients for {st.session_state['cuisine']}")
                    if retrieved_docs:
                        documents_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                        rag_prompt_str = get_safe_prompt(rag_prompt_template, {"cuisine": st.session_state["cuisine"], "documents": documents_text})
                        st.session_state["generated_ingredients"] = llm.predict(rag_prompt_str)
                    else:
                        summary = wikipedia.summary(f"{st.session_state['cuisine']} cuisine", sentences=3, auto_suggest=False)
                        max_tokens_summary = 1000
                        if len(encoding.encode(summary)) > max_tokens_summary:
                            summary = encoding.decode(encoding.encode(summary)[:max_tokens_summary])
                        prompt_str = get_safe_prompt(wikipedia_prompt_template, {"cuisine": st.session_state["cuisine"], "summary": summary})
                        st.session_state["generated_ingredients"] = llm.predict(prompt_str)
                except Exception as e:
                    st.session_state["generated_ingredients"] = f"Error getting ingredients: {e}"

            if st.session_state["show_restaurants"]:
                try:
                    top_restaurants_prompt_str = get_safe_prompt(restaurant_prompt_template, {"cuisine": st.session_state["cuisine"]})
                    st.session_state["generated_restaurants"] = llm.predict(top_restaurants_prompt_str)
                except Exception as e:
                    st.session_state["generated_restaurants"] = f"Error getting restaurants: {e}"

        st.session_state["stage"] = "final_display"
        st.rerun()
    
    if st.button("Skip and Start Chat"):
        st.session_state["stage"] = "final_display"
        st.rerun()

# --- Stage 5: Final Display and Chat ---
elif st.session_state["stage"] == "final_display":
    display_final_content()

    with st.sidebar:
        st.subheader("💬 Cuisine Chat")
        if st.session_state["cuisine"]:
            st.info("Ask anything about " + st.session_state["cuisine"] + " cuisine!")
        else:
            st.info("Start a new session to begin chatting.")

        user_input = st.text_input("Your question:", key="chat_input")

        if st.button("Send"):
            if user_input:
                if "chat_history" not in st.session_state:
                    st.session_state["chat_history"] = []

                chat_context = ""
                for role, msg in st.session_state["chat_history"][-4:]:
                    chat_context += f"{role}: {msg}\n"
                
                prompt = f"""You are a helpful assistant specializing in world cuisines.
                The user is currently interested in {st.session_state['cuisine']} cuisine.
                All of your responses should be about this specific cuisine, unless the user explicitly asks about something else.

                Chat History:
                {chat_context}
                User: {user_input}
                AI:"""

                st.session_state["chat_history"].append(("User", user_input))
                
                with st.spinner("Thinking..."):
                    try:
                        response = llm.predict(prompt)
                        st.session_state["chat_history"].append(("AI", response))
                    except Exception as e:
                        st.error(f"Error generating chat response: {e}")
                        st.session_state["chat_history"].append(("AI", "Sorry, an error occurred. Please try again."))
        
        st.markdown("---")
        
        if "chat_history" in st.session_state:
            for i in range(len(st.session_state["chat_history"]) - 1, -1, -2):
                if i - 1 >= 0:
                    user_role, user_msg = st.session_state["chat_history"][i-1]
                    ai_role, ai_msg = st.session_state["chat_history"][i]
                    
                    st.markdown(f"**👤 You:** {user_msg}")
                    st.markdown(f"**🤖 AI:** {ai_msg}")
                    
                    st.markdown("---")

        if st.button("Clear Chat"):
            if "chat_history" in st.session_state:
                del st.session_state["chat_history"]
            st.rerun()

        st.markdown("---")
        if st.button("Start New Cuisine"):
            reset_session_state()
            st.rerun()