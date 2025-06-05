import os
import re
import json
import logging
import httpx
import pandas as pd
import numpy as np
from typing import Dict, List, TypedDict, Annotated, Literal
from fastapi import HTTPException
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma3:4b")

# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The conversation messages"]
    prompt: str
    dataframe: pd.DataFrame
    columns: List[str]
    intent: Dict
    chart_config: Dict
    code: str
    result: Dict
    error: str
    next_action: str

async def generate_with_ollama(prompt, temperature=0.2):
    """Generate response using Ollama with the specified model."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.95,
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
    except Exception as e:
        logger.error(f"Error generating response with Ollama: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response with Ollama: {str(e)}")

# Agent nodes
async def analyze_intent_node(state: AgentState) -> AgentState:
    """Analyze the user's prompt to determine intent."""
    prompt = state["prompt"]
    columns = state["columns"]
    
    response_format = {
        "intent": "statistical",
        "reason": "Prompt requests statistical analysis",
        "visualization_type": None,
        "transformation_type": None,
        "statistical_type": "correlation"
    }

    input_text = f"""Analyze the following prompt and determine if it's requesting data transformation, visualization, or statistical analysis:

Prompt: {prompt}
Available columns: {', '.join(columns)}

Provide a JSON response with:
1. intent: Either 'visualization', 'transformation', or 'statistical'
2. reason: Brief explanation of why this classification was chosen
3. visualization_type: If intent is 'visualization', specify the chart type ('bar', 'line', 'pie', 'scatter', 'area', 'histogram', 'boxplot')
4. transformation_type: If intent is 'transformation', specify the operation type ('aggregate', 'filter', 'join', 'compute', 'sort', 'group')
5. statistical_type: If intent is 'statistical', specify the test type ('correlation', 'ttest', 'regression', 'descriptive'), 

Example response format:
{json.dumps(response_format)}"""

    try:
        json_text = await generate_with_ollama(input_text, temperature=0.4)
        
        # Try to extract JSON from markdown code blocks if present
        json_match = re.search(r"```(?:json)?\n(.*?)\n```", json_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        
        json_text = json_text.strip()
        
        try:
            intent = json.loads(json_text)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract just the JSON object
            json_obj_match = re.search(r"(\{.*\})", json_text, re.DOTALL)
            if json_obj_match:
                intent = json.loads(json_obj_match.group(1))
            else:
                raise
        
        state["intent"] = intent
        state["next_action"] = intent["intent"]
        logger.info(f"Intent analysis result: {intent}")
        
    except Exception as e:
        state["error"] = f"Error analyzing prompt intent: {str(e)}"
        state["next_action"] = "error"
        logger.error(f"Error in analyze_intent_node: {str(e)}")
    
    return state

async def generate_visualization_node(state: AgentState) -> AgentState:
    """Generate visualization configuration and code."""
    prompt = state["prompt"]
    columns = state["columns"]
    
    response_format = {
        "chart_type": "bar",
        "x_axis": "date",
        "y_axis": "sales",
        "aggregation": "sum",
        "title": "Total Sales by Date"
    }
    
    input_text = f"""Based on the following prompt, determine the appropriate chart configuration:

Prompt: {prompt}
Available columns: {', '.join(columns)}

Generate a JSON configuration with:
1. chart_type: 'bar', 'line', 'pie', 'scatter', 'area', 'histogram', 'boxplot'
2. x_axis: column name for x-axis
3. y_axis: column name for y-axis (can be None for histograms)
4. aggregation: 'sum', 'mean', 'count', 'none'
5. title: chart title

Example response format:
{json.dumps(response_format)}

Provide only the JSON configuration, no explanations."""

    try:
        json_text = await generate_with_ollama(input_text, temperature=0.5)
        
        json_match = re.search(r"```(?:json)?\n(.*?)\n```", json_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        
        json_text = json_text.strip()
        
        try:
            chart_config = json.loads(json_text)
        except json.JSONDecodeError:
            json_obj_match = re.search(r"(\{.*\})", json_text, re.DOTALL)
            if json_obj_match:
                chart_config = json.loads(json_obj_match.group(1))
            else:
                raise
                
        state["chart_config"] = chart_config
        state["next_action"] = "execute"
        logger.info(f"Generated chart config: {chart_config}")
        
    except Exception as e:
        state["error"] = f"Error generating chart configuration: {str(e)}"
        state["next_action"] = "error"
        logger.error(f"Error in generate_visualization_node: {str(e)}")
    
    return state

async def generate_transformation_node(state: AgentState) -> AgentState:
    """Generate pandas transformation code."""
    prompt = state["prompt"]
    columns = state["columns"]
    
    input_text = f"""Write Python code to perform the following pandas DataFrame transformation:

{prompt}

Available columns: {', '.join(columns)}

Pandas Knowledge Base:
1. DataFrame Operations:
   - select columns: df[['col1', 'col2']]
   - filter rows: df[df['column'] > value]
   - group data: df.groupby('column')
   - sort data: df.sort_values('column')
   - add/modify columns: df['new_col'] = df['col1'] * 2
   - drop columns: df.drop(['col1'], axis=1)
   - remove duplicates: df.drop_duplicates()
   - merge dataframes: pd.merge(df1, df2)

2. Common Functions:
   - df.apply(): Apply function to columns/rows
   - df.fillna(): Fill missing values
   - df.dropna(): Drop missing values
   - df.replace(): Replace values
   - pd.to_datetime(): Convert to datetime
   - df.astype(): Convert data types
   - df.round(): Round numbers
   - df.sum(), df.mean(), df.count(): Aggregations

3. String Operations:
   - df['col'].str.contains(): String contains
   - df['col'].str.split(): Split strings
   - df['col'].str.replace(): Replace in strings
   - df['col'].str.upper(): Convert to uppercase

4. Window Operations:
   - df.rolling(): Rolling window operations
   - df.shift(): Shift values
   - df.expanding(): Expanding window

Requirements:
1. Use pandas DataFrame operations
2. Handle missing values appropriately
3. Store result in 'transformed_df'
4. DO NOT define functions
5. Return a pandas DataFrame
6. Use proper type conversions if needed

Available variables:
- df: pandas DataFrame
- pd: pandas module
- np: numpy module

Example format:
```python
transformed_df = df.copy()
transformed_df['new_column'] = df['column1'] * df['column2']
transformed_df = transformed_df.fillna(0)  # Handle nulls
```

Provide only the code, no explanations. DO NOT DEFINE functions, directly perform the operations on the df."""

    try:
        code = await generate_with_ollama(input_text, temperature=0.4)
        
        code_match = re.search(r"```python\n(.*?)\n```", code, re.DOTALL)
        code = code_match.group(1) if code_match else code
        
        state["code"] = code
        state["next_action"] = "execute"
        logger.info(f"Generated transformation code: {code}")
        
    except Exception as e:
        state["error"] = f"Error generating transformation code: {str(e)}"
        state["next_action"] = "error"
        logger.error(f"Error in generate_transformation_node: {str(e)}")
    
    return state

async def generate_statistical_node(state: AgentState) -> AgentState:
    """Generate pandas/numpy code for statistical analysis."""
    prompt = state["prompt"]
    columns = state["columns"]
    
    input_text = f"""Write pandas/numpy code to perform the following statistical analysis:

{prompt}

Available columns: {', '.join(columns)}

Statistical Analysis Knowledge Base:
1. Descriptive Statistics:
   - df.describe(): Summary statistics
   - df.mean(), df.std(): Mean and standard deviation
   - df.var(): Variance
   - df.min(), df.max(): Min/max values
   - df.quantile([0.25, 0.5, 0.75]): Quartiles
   - df.corr(): Correlation matrix

2. Hypothesis Testing (scipy.stats):
   - stats.ttest_ind(): Independent t-test
   - stats.ttest_rel(): Paired t-test
   - stats.chi2_contingency(): Chi-square test
   - stats.pearsonr(): Pearson correlation
   - stats.spearmanr(): Spearman correlation

3. Regression Analysis:
   - np.polyfit(): Polynomial fitting
   - stats.linregress(): Linear regression
   - df.rolling().corr(): Rolling correlation

4. Data Quality:
   - df.isnull().sum(): Count missing values
   - df.duplicated().sum(): Count duplicates
   - df.value_counts(): Value counts

Requirements:
1. Use pandas and numpy functions
2. Include proper statistical computations
3. Store main result in 'stat_result'
4. DO NOT define functions
5. Handle null values appropriately
6. Include interpretation comments

Available variables:
- df: pandas DataFrame
- pd: pandas module
- np: numpy module
- stats: scipy.stats module

Example formats:

For correlation analysis:
```python
# Calculate correlation matrix
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
stat_result = correlation_matrix
```

For descriptive statistics:
```python
# Generate summary statistics
stat_result = df.describe()
# Add correlation for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    stat_result = {
        'descriptive': df.describe(),
        'correlation': df[numeric_cols].corr()
    }
```

For hypothesis testing:
```python
# Perform t-test between two groups
group1 = df[df['category'] == 'A']['value']
group2 = df[df['category'] == 'B']['value']
t_stat, p_value = stats.ttest_ind(group1, group2)
stat_result = {'t_statistic': t_stat, 'p_value': p_value}
```

Provide only the code, no explanations. DO NOT DEFINE functions."""

    try:
        code = await generate_with_ollama(input_text, temperature=0.3)
        
        code_match = re.search(r"```python\n(.*?)\n```", code, re.DOTALL)
        code = code_match.group(1) if code_match else code
        
        state["code"] = code
        state["next_action"] = "execute"
        logger.info(f"Generated statistical code: {code}")
        
    except Exception as e:
        state["error"] = f"Error generating statistical code: {str(e)}"
        state["next_action"] = "error"
        logger.error(f"Error in generate_statistical_node: {str(e)}")
    
    return state

async def execute_code_node(state: AgentState) -> AgentState:
    """Execute the generated code safely."""
    code = state["code"]
    df = state["dataframe"]
    
    if not code:
        state["error"] = "No code to execute"
        state["next_action"] = "error"
        return state
    
    try:
        # Create safe execution environment
        safe_globals = {
            'df': df,
            'pd': pd,
            'np': np,
            'stats': stats,
            'plt': plt,
            'sns': sns
        }
        
        # Execute the code
        exec(code, safe_globals)
        
        # Extract results based on intent
        intent = state["intent"]["intent"]
        
        if intent == "transformation":
            if 'transformed_df' in safe_globals:
                result_df = safe_globals['transformed_df']
                state["result"] = {
                    "type": "transformation",
                    "shape": result_df.shape,
                    "columns": result_df.columns.tolist(),
                    "head": result_df.head().to_dict(),
                    "dataframe": result_df
                }
            else:
                state["error"] = "No 'transformed_df' found in execution result"
                
        elif intent == "statistical":
            if 'stat_result' in safe_globals:
                stat_result = safe_globals['stat_result']
                if isinstance(stat_result, pd.DataFrame):
                    state["result"] = {
                        "type": "statistical",
                        "data": stat_result.to_dict(),
                        "shape": stat_result.shape
                    }
                elif isinstance(stat_result, dict):
                    # Convert any DataFrames in dict to serializable format
                    serializable_result = {}
                    for key, value in stat_result.items():
                        if isinstance(value, pd.DataFrame):
                            serializable_result[key] = value.to_dict()
                        else:
                            serializable_result[key] = value
                    state["result"] = {
                        "type": "statistical",
                        "data": serializable_result
                    }
                else:
                    state["result"] = {
                        "type": "statistical", 
                        "data": stat_result
                    }
            else:
                state["error"] = "No 'stat_result' found in execution result"
                
        elif intent == "visualization":
            # For visualization, we would typically save the plot and return path
            chart_config = state.get("chart_config", {})
            chart_type = chart_config.get("chart_type", "bar")
            
            state["result"] = {
                "type": "visualization",
                "chart_type": chart_type,
                "config": chart_config,
                "message": "Visualization code executed successfully"
            }
        
        state["next_action"] = "complete"
        logger.info("Code executed successfully")
        
    except Exception as e:
        state["error"] = f"Error executing code: {str(e)}"
        state["next_action"] = "error"
        logger.error(f"Error in execute_code_node: {str(e)}")
    
    return state

async def error_handler_node(state: AgentState) -> AgentState:
    """Handle errors and provide feedback."""
    error = state.get("error", "Unknown error occurred")
    logger.error(f"Error in agent workflow: {error}")
    
    state["result"] = {
        "type": "error",
        "message": error,
        "suggestions": [
            "Check if the column names are correct",
            "Verify that the data types are appropriate",
            "Ensure the prompt is clear and specific"
        ]
    }
    state["next_action"] = "complete"
    return state

def route_based_on_intent(state: AgentState) -> Literal["visualization", "transformation", "statistical", "error"]:
    """Route to appropriate node based on intent analysis."""
    if state.get("error"):
        return "error"
    
    intent = state.get("intent", {}).get("intent", "error")
    return intent

def route_to_execution(state: AgentState) -> Literal["execute", "error", "complete"]:
    """Route to execution or error handling."""
    if state.get("error"):
        return "error"
    
    next_action = state.get("next_action", "error")
    if next_action == "execute":
        return "execute"
    elif next_action == "complete":
        return "complete"
    else:
        return "error"

# Build the LangGraph workflow
def create_data_analysis_agent():
    """Create the data analysis agent using LangGraph."""
    
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("analyze_intent", analyze_intent_node)
    workflow.add_node("visualization", generate_visualization_node)
    workflow.add_node("transformation", generate_transformation_node)
    workflow.add_node("statistical", generate_statistical_node)
    workflow.add_node("execute", execute_code_node)
    workflow.add_node("error_handler", error_handler_node)
    
    # Add edges
    workflow.add_edge(START, "analyze_intent")
    
    # Conditional edges based on intent
    workflow.add_conditional_edges(
        "analyze_intent",
        route_based_on_intent,
        {
            "visualization": "visualization",
            "transformation": "transformation", 
            "statistical": "statistical",
            "error": "error_handler"
        }
    )
    
    # Route from generation nodes to execution
    workflow.add_conditional_edges(
        "visualization",
        route_to_execution,
        {
            "execute": "execute",
            "complete": END,
            "error": "error_handler"
        }
    )
    workflow.add_conditional_edges(
        "transformation",
        route_to_execution,
        {
            "execute": "execute",
            "complete": END,
            "error": "error_handler"
        }
    )
    
    # Final edges
    workflow.add_edge("execute", END)
    workflow.add_edge("error_handler", END)
    
    # Compile the graph
    app = workflow.compile()
    return app

# Main execution function
async def analyze_data_with_agent(prompt: str, dataframe: pd.DataFrame) -> Dict:
    """
    Analyze data using the LangGraph agent.
    
    Args:
        prompt: Natural language prompt describing the analysis
        dataframe: Pandas DataFrame to analyze
        
    Returns:
        Dictionary containing the analysis results
    """
    # Create the agent
    agent = create_data_analysis_agent()
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=prompt)],
        "prompt": prompt,
        "dataframe": dataframe,
        "columns": dataframe.columns.tolist(),
        "intent": {},
        "chart_config": {},
        "code": "",
        "result": {},
        "error": "",
        "next_action": ""
    }
    
    # Run the agent
    try:
        final_state = await agent.ainvoke(initial_state)
        return final_state["result"]
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        return {
            "type": "error",
            "message": f"Agent execution failed: {str(e)}"
        }

# Test function
async def test_agent():
    """Test the data analysis agent."""
    # Create sample data
    data = {
        'date': pd.date_range('2024-01-01', periods=100),
        'sales': np.random.normal(1000, 200, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
    }
    df = pd.DataFrame(data)
    
    # Test different types of prompts
    test_prompts = [
        "Create a bar chart showing average sales by category",
        "Calculate correlation between date and sales",
        "Filter the data to show only category A and add a profit column that is 20% of sales"
    ]
    
    for prompt in test_prompts:
        print(f"\n--- Testing: {prompt} ---")
        result = await analyze_data_with_agent(prompt, df)
        print(f"Result: {result}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agent())