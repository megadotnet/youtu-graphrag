"""
Configuration loader and manager for the Youtu-GraphRAG framework.

This module handles the loading, validation, and access management of configuration parameters
defined in YAML files. It uses data classes to structure the configuration settings for various
components of the framework, such as datasets, triggers, construction, retrieval, and evaluation.
"""

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List
import yaml

from utils.logger import logger


@dataclass
class DatasetConfig:
    """
    Configuration for a specific dataset.

    Attributes:
        corpus_path (str): Path to the corpus file containing the dataset content.
        qa_path (str): Path to the question-answer pairs file for evaluation or training.
        schema_path (str): Path to the schema file defining the graph structure.
        graph_output (str): Path where the constructed knowledge graph will be saved.
    """
    corpus_path: str
    qa_path: str
    schema_path: str
    graph_output: str

@dataclass
class TriggersConfig:
    """
    Configuration for execution triggers and modes.

    Attributes:
        constructor_trigger (bool): Whether to trigger the knowledge graph construction process. Default is True.
        retrieve_trigger (bool): Whether to trigger the information retrieval process. Default is True.
        mode (str): The operational mode of the system, either "agent" or "noagent". Default is "agent".
    """
    constructor_trigger: bool = True
    retrieve_trigger: bool = True
    mode: str = "agent"  # "agent" or "noagent"

@dataclass
class ConstructionConfig:
    """
    Configuration for knowledge graph construction.

    Attributes:
        mode (str): The construction mode (e.g., "agent"). Default is "agent".
        max_workers (int): Maximum number of worker threads/processes. Default is 32.
        datasets_no_chunk (list): List of datasets that should not be chunked.
        chunk_size (int): Size of text chunks for processing. Default is 1000.
        overlap (int): Overlap size between consecutive chunks. Default is 200.
    """
    mode: str = "agent"
    max_workers: int = 32
    datasets_no_chunk: list = None
    chunk_size: int = 1000
    overlap: int = 200
    
    def __post_init__(self):
        """Initializes default values for datasets_no_chunk if not provided."""
        if self.datasets_no_chunk is None:
            self.datasets_no_chunk = ["hotpot", "2wiki", "musique", "graphrag-bench", "anony_chs", "anony_eng"]

@dataclass
class TreeCommConfig:
    """
    Configuration for the Tree Community detection algorithm.

    Attributes:
        embedding_model (str): The name of the embedding model to use. Default is "all-MiniLM-L6-v2".
        struct_weight (float): Weight for structural information in community detection. Default is 0.3.
        enable_fast_mode (bool): Whether to enable fast execution mode. Default is True.
        max_total_communities (int): Maximum total number of communities to detect. Default is 100.
    """
    embedding_model: str = "all-MiniLM-L6-v2"
    struct_weight: float = 0.3
    enable_fast_mode: bool = True
    max_total_communities: int = 100

@dataclass
class FAISSConfig:
    """
    Configuration for FAISS (Facebook AI Similarity Search).

    Attributes:
        search_k (int): Number of nearest neighbors to search for. Default is 50.
        max_workers (int): Maximum number of workers for parallel processing. Default is 4.
        device (str): Device to run FAISS on ("cpu" or "cuda"). Default is "cpu".
    """
    search_k: int = 50
    max_workers: int = 4
    device: str = "cpu"

@dataclass
class AgentConfig:
    """
    Configuration for Agent-based retrieval mode.

    Attributes:
        max_steps (int): Maximum number of reasoning steps. Default is 5.
        enable_ircot (bool): Whether to enable Iterative Retrieval Chain of Thought (IRCoT). Default is True.
        enable_parallel_subquestions (bool): Whether to process sub-questions in parallel. Default is True.
    """
    max_steps: int = 5
    enable_ircot: bool = True
    enable_parallel_subquestions: bool = True

@dataclass
class RetrievalConfig:
    """
    Configuration for information retrieval.

    Attributes:
        top_k (int): Number of top results to initially retrieve. Default is 5.
        recall_paths (int): Number of recall paths to consider. Default is 2.
        top_k_filter (int): Number of top results to keep after filtering. Default is 20.
        similarity_threshold (float): Threshold for similarity matching. Default is 0.3.
        enable_query_enhancement (bool): Whether to enable query enhancement. Default is True.
        enable_reranking (bool): Whether to enable reranking of results. Default is True.
        enable_high_recall (bool): Whether to enable high recall mode. Default is True.
        enable_caching (bool): Whether to enable caching of retrieval results. Default is True.
        cache_dir (str): Directory to store cache files. Default is "retriever/faiss_cache_new".
        faiss (FAISSConfig): FAISS specific configuration.
        agent (AgentConfig): Agent specific configuration.
    """
    top_k: int = 5
    recall_paths: int = 2
    top_k_filter: int = 20
    similarity_threshold: float = 0.3
    enable_query_enhancement: bool = True
    enable_reranking: bool = True
    enable_high_recall: bool = True
    enable_caching: bool = True
    cache_dir: str = "retriever/faiss_cache_new"
    faiss: FAISSConfig = None
    agent: AgentConfig = None
    
    def __post_init__(self):
        """Initializes nested configuration objects if they are None."""
        if self.faiss is None:
            self.faiss = FAISSConfig()
        if self.agent is None:
            self.agent = AgentConfig()

@dataclass
class EmbeddingsConfig:
    """
    Configuration for text embeddings.

    Attributes:
        model_name (str): Name of the model used for generating embeddings. Default is "all-MiniLM-L6-v2".
        device (str): Device to run the embedding model on. Default is "cpu".
        batch_size (int): Batch size for generating embeddings. Default is 32.
        max_length (int): Maximum length of input text. Default is 512.
    """
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    max_length: int = 512

@dataclass
class NLPConfig:
    """
    Configuration for Natural Language Processing tools.

    Attributes:
        spacy_model (str): Name of the spaCy model to use. Default is 'en_core_web_lg'.
    """
    spacy_model: str = 'en_core_web_lg' 


@dataclass
class OutputConfig:
    """
    Configuration for output directories and files.

    Attributes:
        base_dir (str): Base directory for all outputs. Default is "output".
        graphs_dir (str): Directory for saving graph files. Default is "output/graphs".
        chunks_dir (str): Directory for saving text chunks. Default is "output/chunks".
        logs_dir (str): Directory for saving logs. Default is "output/logs".
        save_intermediate_results (bool): Whether to save intermediate processing results. Default is True.
        save_chunk_details (bool): Whether to save detailed chunk information. Default is True.
    """
    base_dir: str = "output"
    graphs_dir: str = "output/graphs"
    chunks_dir: str = "output/chunks"
    logs_dir: str = "output/logs"
    save_intermediate_results: bool = True
    save_chunk_details: bool = True

@dataclass
class PerformanceConfig:
    """
    Configuration for system performance tuning.

    Attributes:
        parallel_processing (bool): Whether to enable parallel processing. Default is True.
        max_workers (int): Maximum number of workers for parallel tasks. Default is 32.
        batch_size (int): Batch size for processing. Default is 16.
        memory_optimization (bool): Whether to enable memory optimization strategies. Default is True.
    """
    parallel_processing: bool = True
    max_workers: int = 32
    batch_size: int = 16
    memory_optimization: bool = True

@dataclass
class EvaluationConfig:
    """
    Configuration for system evaluation.

    Attributes:
        enable_evaluation (bool): Whether to enable evaluation metrics. Default is True.
        metrics (list): List of metrics to evaluate (e.g., "accuracy", "f1").
        save_detailed_results (bool): Whether to save detailed evaluation results. Default is True.
    """
    enable_evaluation: bool = True
    metrics: list = None
    save_detailed_results: bool = True
    
    def __post_init__(self):
        """Initializes default metrics if not provided."""
        if self.metrics is None:
            self.metrics = ["accuracy", "precision", "recall", "f1"]

class ConfigManager:
    """
    Main configuration manager for the Youtu-GraphRAG framework.

    This class handles loading configuration from a YAML file, validating the values,
    parsing them into structured data classes, and providing methods to access
    configuration settings.

    Attributes:
        config_path (str): Path to the configuration file.
        config_data (Dict[str, Any]): Raw dictionary loaded from the YAML file.
        datasets (Dict[str, DatasetConfig]): Dictionary of dataset configurations.
        triggers (TriggersConfig): Configuration for execution triggers.
        construction (ConstructionConfig): Configuration for graph construction.
        tree_comm (TreeCommConfig): Configuration for community detection.
        retrieval (RetrievalConfig): Configuration for information retrieval.
        embeddings (EmbeddingsConfig): Configuration for embeddings.
        nlp (NLPConfig): Configuration for NLP tools.
        prompts (Dict[str, Any]): Dictionary of prompts used by the system.
        output (OutputConfig): Configuration for output handling.
        performance (PerformanceConfig): Configuration for performance settings.
        evaluation (EvaluationConfig): Configuration for evaluation settings.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path (Optional[str]): Path to the configuration file. If None, the default 'base_config.yaml' is used.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config_data: Dict[str, Any] = {}
        self.datasets: Dict[str, DatasetConfig] = {}
        self.triggers: Optional[TriggersConfig] = None
        self.construction: Optional[ConstructionConfig] = None
        self.tree_comm: Optional[TreeCommConfig] = None
        self.retrieval: Optional[RetrievalConfig] = None
        self.embeddings: Optional[EmbeddingsConfig] = None
        self.nlp: Optional[NLPConfig] = None
        self.prompts: Dict[str, Any] = {}
        self.output: Optional[OutputConfig] = None
        self.performance: Optional[PerformanceConfig] = None
        self.evaluation: Optional[EvaluationConfig] = None

        self.load_config()
    
    def _get_default_config_path(self) -> str:
        """
        Get the default configuration file path.

        Returns:
            str: Absolute path to 'base_config.yaml' located in the same directory as this file.
        """
        current_dir = Path(__file__).parent
        return str(current_dir / "base_config.yaml")
    
    def load_config(self) -> None:
        """
        Load and parse the configuration file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the configuration file contains invalid YAML.
            RuntimeError: If any other error occurs during loading.
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f)
            
            self._parse_config()
            self._validate_config()
            
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")
    
    def _parse_config(self) -> None:
        """
        Parse the loaded configuration data into structured objects.

        Populates the attributes of the ConfigManager instance (datasets, triggers, etc.)
        from the raw config_data dictionary.
        """
        datasets_data = self.config_data.get("datasets", {})
        self.datasets = {
            name: DatasetConfig(**config) 
            for name, config in datasets_data.items()
        }
        
        triggers_data = self.config_data.get("triggers", {})
        self.triggers = TriggersConfig(**triggers_data)
        
        construction_data = self.config_data.get("construction", {})
        tree_comm_data = construction_data.pop("tree_comm", {})
        self.construction = ConstructionConfig(**construction_data)
        self.tree_comm = TreeCommConfig(**tree_comm_data)
        
        retrieval_data = self.config_data.get("retrieval", {})
        faiss_data = retrieval_data.pop("faiss", {})
        agent_data = retrieval_data.pop("agent", {})
        self.retrieval = RetrievalConfig(**retrieval_data)
        self.retrieval.faiss = FAISSConfig(**faiss_data)
        self.retrieval.agent = AgentConfig(**agent_data)
        
        embeddings_data = self.config_data.get("embeddings", {})
        self.embeddings = EmbeddingsConfig(**embeddings_data)
        
        nlp = self.config_data.get("nlp", {})
        self.nlp = NLPConfig(**nlp)
        
        self.prompts = self.config_data.get("prompts", {})
        
        output_data = self.config_data.get("output", {})
        self.output = OutputConfig(**output_data)
        
        performance_data = self.config_data.get("performance", {})
        self.performance = PerformanceConfig(**performance_data)
        
        evaluation_data = self.config_data.get("evaluation", {})
        self.evaluation = EvaluationConfig(**evaluation_data)
    
    def _validate_config(self) -> None:
        """
        Validate the loaded configuration.

        Checks for:
        - Existence of dataset paths (corpus and schema).
        - Validity of operation modes.
        - Logical consistency of numerical parameters.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        for dataset_name, dataset_config in self.datasets.items():
            if not os.path.exists(dataset_config.corpus_path):
                logger.warning(f"Corpus path not found for {dataset_name}: {dataset_config.corpus_path}")
            if not os.path.exists(dataset_config.schema_path):
                logger.warning(f"Schema path not found for {dataset_name}: {dataset_config.schema_path}")
        
        valid_modes = ["agent", "noagent"]
        if self.triggers.mode not in valid_modes:
            raise ValueError(f"Invalid mode: {self.triggers.mode}. Must be one of {valid_modes}")
        
        if self.construction.mode not in valid_modes:
            raise ValueError(f"Invalid construction mode: {self.construction.mode}")
        
        # Validate numerical parameters
        if self.retrieval.top_k <= 0:
            raise ValueError("top_k must be positive")
        
        if self.tree_comm.struct_weight < 0 or self.tree_comm.struct_weight > 1:
            raise ValueError("struct_weight must be between 0 and 1")
    
    def get_dataset_config(self, dataset_name: str) -> DatasetConfig:
        """
        Get configuration for a specific dataset.

        Args:
            dataset_name (str): The name of the dataset.

        Returns:
            DatasetConfig: The configuration object for the requested dataset.

        Raises:
            ValueError: If the dataset name is not found in the configuration.
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found in configuration")
        return self.datasets[dataset_name]
    
    def get_prompt(self, category: str, prompt_type: str) -> str:
        """
        Get a specific prompt template.

        Args:
            category (str): The category of the prompt (e.g., 'retrieval').
            prompt_type (str): The specific type/name of the prompt.

        Returns:
            str: The prompt template string.

        Raises:
            ValueError: If the prompt is not found.
        """
        try:
            return self.prompts[category][prompt_type]
        except KeyError:
            raise ValueError(f"Prompt not found: {category}.{prompt_type}")
    
    def get_prompt_formatted(self, category: str, prompt_type: str, **kwargs) -> str:
        """
        Get a formatted prompt with variables substituted.

        Args:
            category (str): The category of the prompt.
            prompt_type (str): The specific type/name of the prompt.
            **kwargs: Keyword arguments matching the variables in the prompt template.

        Returns:
            str: The formatted prompt string.

        Raises:
            ValueError: If a required variable is missing for the prompt template.
        """
        template = self.get_prompt(category, prompt_type)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing variable {e} for prompt {category}.{prompt_type}")
    
    def override_config(self, overrides: Dict[str, Any]) -> None:
        """
        Override configuration values at runtime.

        Args:
            overrides (Dict[str, Any]): A dictionary containing the values to override.
                Nested keys should be represented as nested dictionaries.
        """
        def update_nested_dict(d: dict, overrides: dict) -> None:
            for key, value in overrides.items():
                if isinstance(value, dict) and key in d and isinstance(d[key], dict):
                    update_nested_dict(d[key], value)
                else:
                    d[key] = value
        
        update_nested_dict(self.config_data, overrides)
        self._parse_config()
        self._validate_config()
    
    def save_config(self, output_path: str) -> None:
        """
        Save current configuration to a file.

        Args:
            output_path (str): The path where the configuration YAML file will be saved.
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config_data, f, default_flow_style=False, ensure_ascii=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format.

        Returns:
            Dict[str, Any]: A dictionary representation of the entire configuration.
        """
        # Note: self.api was referenced in original code but not defined in __init__ or _parse_config.
        # Assuming it's not present or should be handled if added.
        # Removing 'api' from here as it was not in _parse_config.
        return {
            "datasets": {name: asdict(config) for name, config in self.datasets.items()},
            "triggers": asdict(self.triggers),
            "construction": asdict(self.construction),
            "tree_comm": asdict(self.tree_comm),
            "retrieval": asdict(self.retrieval),
            "embeddings": asdict(self.embeddings),
            "prompts": self.prompts,
            "output": asdict(self.output),
            "performance": asdict(self.performance),
            "evaluation": asdict(self.evaluation),
        }
    
    def create_output_directories(self) -> None:
        """
        Create necessary output directories as defined in the configuration.

        Creates base_dir, graphs_dir, chunks_dir, and logs_dir if they don't exist.
        """
        directories = [
            self.output.base_dir,
            self.output.graphs_dir,
            self.output.chunks_dir,
            self.output.logs_dir,
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

_config_instance: Optional[ConfigManager] = None

def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get the global configuration instance.
    
    This function implements a singleton pattern to ensure only one ConfigManager instance exists.

    Args:
        config_path (Optional[str]): Path to configuration file. Only used on the first call.
        
    Returns:
        ConfigManager: The global ConfigManager instance.
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = ConfigManager(config_path)
    
    return _config_instance

def reload_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    Reload the configuration.
    
    Forces a re-initialization of the global ConfigManager instance.

    Args:
        config_path (Optional[str]): Path to configuration file.
        
    Returns:
        ConfigManager: The new ConfigManager instance.
    """
    global _config_instance
    _config_instance = ConfigManager(config_path)
    return _config_instance
