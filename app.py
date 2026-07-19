import streamlit as st
import os
import warnings
from crewai import LLM, Agent, Task, Crew
from crewai.tools import tool
from duckduckgo_search import DDGS

# Clean up console outputs
warnings.filterwarnings("ignore")

# --- UI SETUP ---
st.set_page_config(page_title="AI Content Crew", page_icon="🎬", layout="centered")

st.title("🎬 Multi-Agent Content Planner")
st.write("Activate your autonomous AI research and creative team with a single click.")

# --- SIDEBAR FOR CREDENTIALS ---
st.sidebar.header("🔑 Configuration")
api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password", help="Get a free key from Google AI Studio")

# --- MAIN INTERFACE INPUTS ---
niche_input = st.text_input(
    label="What is your content niche or topic?",
    placeholder="e.g., AI productivity hacks for busy professionals"
)

# --- EXECUTION BUTTON ---
if st.button("🚀 Launch AI Crew", type="primary"):
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar to proceed.")
    elif not niche_input:
        st.warning("Please enter a niche topic for the agents to research.")
    else:
        # Create a visual loading indicator while the agents work
        with st.spinner("🕵️‍♂️ Researcher is analyzing trends... 📝 Creative Director is scripting..."):
            try:
                # Set environment variable dynamically
                os.environ["GOOGLE_API_KEY"] = api_key

                # Configure the Brain
                gemini_llm = LLM(
                    model="gemini/gemini-3.5-flash",
                    api_key=api_key,
                    temperature=0.7
                )

                # Indestructible Search Tool
                @tool("Web Search")
                def web_search(query: str) -> str:
                    """Searches the internet for modern, up-to-date information on a given topic."""
                    try:
                        with DDGS() as ddgs:
                            results = [r for r in ddgs.text(query, max_results=3)]
                            if not results:
                                return "Search returned no results. Rely on internal knowledge."
                            return str(results)
                    except Exception:
                        return "Web search rate-limited. Proceed using internal knowledge."

                # Define Agents
                researcher = Agent(
                    role="Niche Trend Researcher",
                    goal="Identify 3 high-performing, trending topics or search queries in the user's niche.",
                    backstory="You are a data-driven trend analyst. You look up real-time trends and identify high-interest topics.",
                    tools=[web_search],
                    llm=gemini_llm,
                    max_iter=3
                )

                creative_director = Agent(
                    role="Content & Script Director",
                    goal="Take trending topics and transform them into viral content scripts and a structured calendar.",
                    backstory="You are a master of visual storytelling. You know exactly how to structure video scripts to maximize engagement.",
                    llm=gemini_llm
                )

                # Define Tasks
                research_task = Task(
                    description="Research trending topics, common questions, and pain points on the web for the following niche: {niche_topic}.",
                    expected_output="A list of 3 distinct trending topics or sub-themes with brief explanations.",
                    agent=researcher
                )

                creative_task = Task(
                    description="Take the researched trends and create a short content calendar. For each topic, write a 30-second video script containing a Hook, Core Value, and CTA.",
                    expected_output="A formatted content calendar containing topics, catchy titles, visual cue notes, and complete scripts.",
                    agent=creative_director
                )

                # Assemble Crew
                content_crew = Crew(
                    agents=[researcher, creative_director],
                    tasks=[research_task, creative_task]
                )

                # Run Crew (Note: kickoff() works standardly here because regular python files don't suffer from Colab's background event loop conflict)
                crew_output = content_crew.kickoff(inputs={"niche_topic": niche_input})
                
                # --- DISPLAY RESULTS ---
                st.success("✨ Your Content Strategy is Ready!")
                st.markdown("---")
                st.markdown(crew_output.raw)
                
            except Exception as e:
                st.error(f"An unexpected error occurred during execution: {e}")
