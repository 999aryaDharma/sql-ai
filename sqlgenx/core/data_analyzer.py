"""Enhanced data analyzer with token-based limits and cost estimation."""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import json


class TokenEstimator:
    """Estimate token count for cost calculation."""
    
    # Rough approximation: 1 token ≈ 4 characters for English
    CHARS_PER_TOKEN = 4
    
    # Gemini pricing (approximate, check current rates)
    COST_PER_1K_INPUT_TOKENS = 0.00015  # $0.15 per 1M tokens
    COST_PER_1K_OUTPUT_TOKENS = 0.0006  # $0.60 per 1M tokens
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count from text."""
        return len(text) // TokenEstimator.CHARS_PER_TOKEN
    
    @staticmethod
    def estimate_cost(input_tokens: int, output_tokens: int = 500) -> float:
        """Estimate API call cost in USD."""
        input_cost = (input_tokens / 1000) * TokenEstimator.COST_PER_1K_INPUT_TOKENS
        output_cost = (output_tokens / 1000) * TokenEstimator.COST_PER_1K_OUTPUT_TOKENS
        return input_cost + output_cost
    
    @staticmethod
    def estimate_dataframe_tokens(df: pd.DataFrame, max_tokens: int = 50000) -> Tuple[int, bool]:
        """
        Estimate tokens needed for DataFrame.
        Returns: (estimated_tokens, needs_truncation)
        """
        # Sample representation
        sample_text = df.head(10).to_string(index=False)
        sample_tokens = TokenEstimator.estimate_tokens(sample_text)
        
        # Extrapolate for full dataset
        rows_sampled = min(10, len(df))
        estimated_full = (sample_tokens / rows_sampled) * len(df)
        
        # Add overhead for stats and formatting
        estimated_full = int(estimated_full * 1.2)
        
        needs_truncation = estimated_full > max_tokens
        
        return estimated_full, needs_truncation


class DataAnalyzerEnhanced:
    """Enhanced data analyzer with token limits and cost awareness."""
    
    # Default token limits
    MAX_INPUT_TOKENS = 50000  # ~50k tokens for input data
    SOFT_ROW_LIMIT = 1000  # Suggest aggregation above this
    HARD_ROW_LIMIT = 10000  # Force aggregation above this
    
    def __init__(self, api_key: str):
        """Initialize enhanced data analyzer."""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.3,
            max_tokens=2048
        )
        
        self.system_prompt = """You are a data analyst expert.
You will be given query results and the original question.
Your task is to analyze the data and provide clear, actionable insights.

Focus on:
1. Key findings and patterns in the data
2. Notable trends or anomalies
3. Business insights and recommendations
4. Answer the original question directly

Be concise but comprehensive. Use bullet points for clarity.
If the data is empty or insufficient, say so clearly.

IMPORTANT: Base your analysis ONLY on the provided data. Do not make assumptions."""
    
    def analyze_results(
        self,
        df: pd.DataFrame,
        original_query: str,
        sql_query: str,
        metadata: Dict[str, Any],
        force_analyze: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Analyze query results with token-based limits.
        
        Returns: (analysis_text, analysis_metadata)
        """
        
        # Check if analysis is feasible
        estimated_tokens, needs_truncation = TokenEstimator.estimate_dataframe_tokens(
            df, 
            self.MAX_INPUT_TOKENS
        )
        
        analysis_metadata = {
            'estimated_tokens': estimated_tokens,
            'estimated_cost': TokenEstimator.estimate_cost(estimated_tokens),
            'row_count': len(df),
            'truncated': needs_truncation
        }
        
        # Decision logic
        if len(df) == 0:
            return "No data to analyze. The query returned 0 rows.", analysis_metadata
        
        if len(df) > self.HARD_ROW_LIMIT and not force_analyze:
            suggestion = self._suggest_aggregation(df, original_query)
            return suggestion, analysis_metadata
        
        if needs_truncation and not force_analyze:
            warning = f"""⚠️ Dataset is large ({len(df):,} rows, ~{estimated_tokens:,} tokens).

Estimated cost: ${analysis_metadata['estimated_cost']:.4f}

Recommendations:
1. Add LIMIT clause to your query
2. Use GROUP BY to aggregate data
3. Use --force-analyze to proceed anyway

To analyze this dataset:
  sqlgenx execute "your query" --analyze --force-analyze"""
            
            return warning, analysis_metadata
        
        # Prepare data summary
        data_summary = self._prepare_smart_summary(df, metadata, estimated_tokens)
        
        # Build prompt
        prompt = f"""Original Question: {original_query}

SQL Query Executed:
{sql_query}

Data Results:
{data_summary}

Dataset Info:
- Total Rows: {len(df):,}
- Columns: {len(df.columns)}

Please analyze this data and provide insights that answer the original question.
Include key findings, patterns, and actionable recommendations."""
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        # Generate analysis
        response = self.llm.invoke(messages)
        
        return response.content, analysis_metadata
    
    def _prepare_smart_summary(
        self,
        df: pd.DataFrame,
        metadata: Dict[str, Any],
        token_budget: int
    ) -> str:
        """Prepare data summary that fits within token budget."""
        
        summary_parts = []
        
        # Always include: row count and columns
        summary_parts.append(f"Total Rows: {len(df):,}")
        summary_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
        summary_parts.append("")
        
        # For small datasets: include all data
        if len(df) <= 20:
            summary_parts.append("Complete Data:")
            summary_parts.append(df.to_string(index=False, max_rows=20))
            return "\n".join(summary_parts)
        
        # For medium datasets: show sample + stats
        if len(df) <= 100:
            summary_parts.append("First 10 rows:")
            summary_parts.append(df.head(10).to_string(index=False))
            summary_parts.append("")
            summary_parts.append("Last 5 rows:")
            summary_parts.append(df.tail(5).to_string(index=False))
        else:
            # For large datasets: aggressive summarization
            summary_parts.append("Sample (first 5 rows):")
            summary_parts.append(df.head(5).to_string(index=False))
        
        # Add numeric statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary_parts.append("")
            summary_parts.append("Numeric Statistics:")
            stats_df = df[numeric_cols].describe()
            
            # Limit stats columns if too many
            if len(numeric_cols) > 5:
                summary_parts.append(stats_df.iloc[:, :5].to_string())
                summary_parts.append(f"... and {len(numeric_cols) - 5} more columns")
            else:
                summary_parts.append(stats_df.to_string())
        
        # Add categorical distributions (limited)
        categorical_cols = df.select_dtypes(include=['object']).columns
        shown_cats = 0
        for col in categorical_cols:
            if shown_cats >= 3:  # Limit categorical summaries
                break
            
            unique_count = df[col].nunique()
            if unique_count <= 10:
                summary_parts.append("")
                summary_parts.append(f"{col} Distribution:")
                value_counts = df[col].value_counts().head(5)
                summary_parts.append(value_counts.to_string())
                shown_cats += 1
        
        return "\n".join(summary_parts)
    
    def _suggest_aggregation(self, df: pd.DataFrame, original_query: str) -> str:
        """Suggest how to aggregate large dataset."""
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        suggestions = [
            f"⚠️ Dataset too large for analysis ({len(df):,} rows)",
            "",
            "Suggested approaches:",
            ""
        ]
        
        # Suggest aggregation queries
        if categorical_cols and numeric_cols:
            suggestions.append("1. Aggregate by category:")
            suggestions.append(f"   SELECT {categorical_cols[0]}, ")
            suggestions.append(f"          AVG({numeric_cols[0]}) as avg_value,")
            suggestions.append(f"          COUNT(*) as count")
            suggestions.append(f"   FROM your_table")
            suggestions.append(f"   GROUP BY {categorical_cols[0]}")
            suggestions.append("")
        
        if numeric_cols:
            suggestions.append("2. Get summary statistics:")
            suggestions.append(f"   SELECT ")
            suggestions.append(f"          MIN({numeric_cols[0]}) as min_value,")
            suggestions.append(f"          MAX({numeric_cols[0]}) as max_value,")
            suggestions.append(f"          AVG({numeric_cols[0]}) as avg_value")
            suggestions.append(f"   FROM your_table")
            suggestions.append("")
        
        suggestions.append("3. Add LIMIT clause:")
        suggestions.append(f"   ... LIMIT 1000")
        suggestions.append("")
        suggestions.append("Or use --force-analyze to proceed (may be expensive)")
        
        return "\n".join(suggestions)
    
    def estimate_analysis_cost(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Estimate cost of analyzing a DataFrame."""
        
        estimated_tokens, needs_truncation = TokenEstimator.estimate_dataframe_tokens(df)
        estimated_cost = TokenEstimator.estimate_cost(estimated_tokens)
        
        return {
            'estimated_input_tokens': estimated_tokens,
            'estimated_cost_usd': estimated_cost,
            'row_count': len(df),
            'recommended': len(df) <= self.SOFT_ROW_LIMIT,
            'needs_truncation': needs_truncation
        }