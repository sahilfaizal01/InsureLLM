import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.utilities.semanticscholar import SemanticScholarAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

load_dotenv()

class ResearchAssistant:
    def __init__(self, vector_store=None):
        """
        Initialize Research Assistant
        Args:
            vector_store (PaperVectorStore, optional): Vector database for storing papers
        """
        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4-turbo",
            api_key=os.getenv('OPENAI_API_KEY')
        )

        # Initialize Semantic Scholar Wrapper
        api_wrapper = SemanticScholarAPIWrapper(
            doc_content_chars_max=1000,
            top_k_results=5
        )

        # Tools
        self.tools = [SemanticScholarQueryRun(api_wrapper=api_wrapper)]
        
        # Conversation Memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input"
        )

        # Vector Store (optional)
        self.vector_store = vector_store
        
        # Prompt Template with citation instructions
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert research assistant. Help the user find and understand scholarly information. Always provide citations to papers you reference using [Author, Year] format and include a references section at the end of your response."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # Create Agent
        self.agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Agent Executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            memory=self.memory
        )
        
        # Track fetched papers for the current session
        self.current_papers = []

    def query(self, input_text):
        """
        Process research query
        Args:
            input_text (str): User's research query
        Returns:
            dict: Research response and fetched papers info
        """
        response = self.agent_executor.invoke({"input": input_text})
        
        # Extract papers from response
        papers = self._extract_papers(response)
        papers_added = False
        
        # Store papers in vector database if available
        if self.vector_store and papers:
            self.vector_store.add_papers(papers)
            self.current_papers = papers
            papers_added = True
            
        return {
            'output': response['output'],
            'papers_added': papers_added,
            'paper_count': len(papers),
            'papers': papers
        }
        
    def _extract_papers(self, response):
        """
        Extract papers from Semantic Scholar response
        Args:
            response (dict): Agent response
        Returns:
            List of paper dictionaries
        """
        papers = []
        
        # Check if tool outputs exist in the response
        if 'intermediate_steps' in response:
            for step in response['intermediate_steps']:
                # Check if the step contains tool output
                if len(step) > 1 and isinstance(step[1], dict) and 'papers' in step[1]:
                    for paper in step[1]['papers']:
                        papers.append({
                            'title': paper.get('title', 'Unknown Title'),
                            'authors': [author.get('name', 'Unknown') for author in paper.get('authors', [])],
                            'abstract': paper.get('abstract', 'No abstract available'),
                            'year': paper.get('year', 'Unknown'),
                            'url': paper.get('url', ''),
                            'venue': paper.get('venue', 'Unknown'),
                            'citation_count': paper.get('citationCount', 0)
                        })
        
        return papers

    def format_response_with_citations(self, response_text, papers):
        """
        Format the response with citations and references
        Args:
            response_text (str): Original response text
            papers (list): List of papers to cite
        Returns:
            str: Formatted response with references
        """
        # Add references section
        if papers:
            references = "\n\n## References\n"
            for i, paper in enumerate(papers):
                authors = ", ".join(paper['authors'][:3])
                if len(paper['authors']) > 3:
                    authors += " et al."
                references += f"{i+1}. {authors} ({paper['year']}). {paper['title']}. *{paper['venue']}*.\n"
            
            return response_text + references
        
        return response_text
