import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document

# Keep all the Pydantic models the same
class ServiceDetail(BaseModel):
    name: str = Field(default="", description="Name of the service or item")
    description: str = Field(default="", description="Detailed description")

class RenewablePODetail(BaseModel):
    name: str = Field(default="")
    description: str = Field(default="")
    duration: str = Field(default="")

class ModificationOption(BaseModel):
    name: str = Field(default="")
    description: str = Field(default="")
    impact_on_revenue: str = Field(default="")

class VariableConsideration(BaseModel):
    name: str = Field(default="")
    amount: str = Field(default="")
    estimation: str = Field(default="")

class FinancingComponent(BaseModel):
    name: str = Field(default="")
    considerations: str = Field(default="")

class AdditionalCost(BaseModel):
    cost: str = Field(default="")
    amount: str = Field(default="")
    description: str = Field(default="")

class ReimbursableExpense(BaseModel):
    expense: str = Field(default="")
    amount: str = Field(default="")
    description: str = Field(default="")

class NonRecoverableCost(BaseModel):
    name: str = Field(default="")
    amount: str = Field(default="")
    description: str = Field(default="")

class CompensationStructure(BaseModel):
    name: str = Field(default="")
    description: str = Field(default="")

class TransactionPriceDetails(BaseModel):
    total_transaction_price: Dict[str, str] = Field(
        default_factory=lambda: {
            "amount_applicable_on_contract_date_including_variable_considerations": ""
        }
    )
    compensation_structure: Optional[CompensationStructure] = Field(default=None)
    variable_considerations: List[VariableConsideration] = Field(default_factory=list)
    financing_components_non_cash_considerations: List[FinancingComponent] = Field(
        default_factory=list
    )

class ContractAnalysisModel(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        validate_assignment=True
    )

    renewable_POs: Dict[str, Union[bool, List[RenewablePODetail]]] = Field(
        default_factory=lambda: {
            "availability": "",
            "details": []
        }
    )
    
    multiple_contracts_with_customer: Dict[str, Union[bool, Dict[str, str]]] = Field(
        default_factory=lambda: {
            "yes/no": False,
            "should_they_be_combined_as_one_contract": {
                "rationale": ""
            }
        }
    )
    
    clarity_of_terms_payment_terms: Dict[str, Union[str, List[ServiceDetail]]] = Field(
        default_factory=lambda: {
            "clarity_of_terms": "",
            "areas_lacking_clarity": []
        }
    )
    
    performance_obligations: Dict[str, Union[str, List[ServiceDetail]]] = Field(
        default_factory=lambda: {
            "distinct_services_promised": "",
            "services_included": []
        }
    )
    
    services_distinct_in_nature_as_seperate_POs: Dict[str, Union[bool, List[ServiceDetail]]] = Field(
        default_factory=lambda: {
            "yes/no": False,
            "how_service_meet_criterion": []
        }
    )
    
    services_distinct_in_terms_of_contract: Dict[str, Union[bool, List[ServiceDetail]]] = Field(
        default_factory=lambda: {
            "yes/no": False,
            "details": []
        }
    )
    
    contract_modifications_or_variable_considerations_affecting_PO: Dict[str, Union[bool, List[ModificationOption]]] = Field(
        default_factory=lambda: {
            "yes/no": False,
            "modifications_options": []
        }
    )
    
    transaction_price: TransactionPriceDetails = Field(default_factory=TransactionPriceDetails)
    
    other_costs_associated_with_delivery: Dict[str, Union[bool, List[AdditionalCost]]] = Field(
        default_factory=lambda: {
            "yes/no": False,
            "details": []
        }
    )
    
    reimbursable_expenses_billed_to_client_separately: Dict[str, Union[bool, ReimbursableExpense]] = Field(
        default_factory=lambda: {
            "available_yes/no": False,
            "details": {}
        }
    )
    
    non_recoverable_costs: Dict[str, Union[bool, List[NonRecoverableCost]]] = Field(
        default_factory=lambda: {
            "yes/no": False,
            "details": []
        }
    )

class ContractRAGSystem:
    def __init__(self, model_name=None):
        # Load environment variables
        load_dotenv()
        
        # Setup logging
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('contract_rag.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Get model name from environment if not provided
        self.model_name = model_name or os.getenv('MODEL_NAME', 'meta-llama/Llama-3.2-1B')
        
        # Get Hugging Face token from environment
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not self.hf_token:
            self.logger.warning("No Hugging Face token found in environment")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype="auto",
            token=self.hf_token
        )
        
        # Create text generation pipeline
        temperature = float(os.getenv('MODEL_TEMPERATURE', '0.1'))
        max_tokens = int(os.getenv('MAX_NEW_TOKENS', '32000'))
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=temperature,
            max_new_tokens=max_tokens
        )
        
        # Output directory from environment
        self.output_dir = Path(os.getenv('FINAL_OUTPUT_DIR', 'final_extracted_data'))
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize embeddings with model from environment
        embedding_model = os.getenv('EMBEDDING_MODEL', 'intfloat/e5-large-v2')
        
        # Fix: Initialize embeddings without passing token directly
        os.environ["HUGGINGFACE_API_KEY"] = self.hf_token  # Set token as environment variable
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )
        
        # Text splitter configuration from environment
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv('CHUNK_SIZE', '4000')),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '200')),
            length_function=len
        )

        
    def load_contract_documents(self, contract_dir: Path):
        """Load contract documents with metadata and content"""
        markdown_path = contract_dir / 'contract.md'
        
        try:
            with open(markdown_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.logger.error(f"Error reading markdown: {e}")
            content = ""
        
        full_text = f"Contract Content:\n{content}"
        
        document = Document(
            page_content=full_text,
            metadata={"source": str(markdown_path)}
        )
        
        return [document]

    def create_vector_store(self, documents):
        """Create FAISS vector store from documents"""
        texts = self.text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(texts, self.embeddings)
        return vector_store

    def query_model(self, context: str, query: str) -> Dict[str, Any]:
        # The comprehensive prompt template
        prompt_template = """
        You are an advanced contract analysis AI with expertise in financial reporting standards (IFRS 15). 
        Your task is to meticulously extract and interpret all the contract details with extreme precision and logical reasoning according to the json schema provided and with reference to the instruction provided in every particular field.
        For any missing information, interpret things by your side, referring to the instruction present inside the json structure of that field.

        Analytical Guidelines:
        1. Be EXHAUSTIVE in your analysis
        2. Infer implicit information logically
        3. Provide EVIDENCE for each interpretation
        4. Use HIERARCHICAL and STRUCTURED responses
        5. CROSS-REFERENCE multiple contract sections
        6. HIGHLIGHT potential ambiguities or risks

        Extraction Rules:
        - Extract ONLY specified details
        - If information is partially available, EXPLAIN the gaps
        - Use probabilistic language for uncertain interpretations
        - Provide CONTEXT for each extracted detail
        - LIST all the information related UNDERSTANDING the field context.
        - Maintain OBJECTIVITY

        CRITICAL THINKING FRAMEWORK:
        - What is EXPLICITLY stated?
        - What can be LOGICALLY INFERRED?
        - What POTENTIAL RISKS or LIMITATIONS exist?
        - How do different contract sections INTERRELATE?

        Context: {context}
        Specific Query: {query}

        RESPONSE INSTRUCTIONS:
        - JSON format MUST be followed exactly
        - Include confidence levels where applicable
        - Explain reasoning behind each interpretation
        - Flag any potential compliance or interpretational challenges
        - Must be data-rich
        - Include Yes/No with brief details where applicable
        - Some information might not be explicitly stated, but must be precisely inferred from the context.

        Response in JSON format:
        """
        
        try:
            # Generate response using HuggingFace pipeline
            full_prompt = prompt_template.format(context=context, query=query)
            response = self.pipe(full_prompt)[0]['generated_text']
            
            # Extract JSON portion from response
            try:
                # Find the start of the JSON content
                json_start = response.find('{')
                if json_start == -1:
                    self.logger.error("No JSON content found in response")
                    return {}
                
                json_content = response[json_start:]
                
                # Parse and validate with Pydantic
                parsed_content = ContractAnalysisModel.model_validate_json(json_content)
                self.logger.info("Successfully parsed and validated JSON")
                return parsed_content.model_dump()
                
            except Exception as e:
                self.logger.error(f"Validation error: {str(e)}")
                self.logger.error(f"Raw content that failed: {response}")
                return {"error": "Failed to validate response", "raw_content": response}
                
        except Exception as e:
            self.logger.error(f"Error querying model: {str(e)}")
            return {"error": str(e)}

    def process_contracts(self, extracted_data_dir: str = "extracted_data"):
        """Process all contracts using RAG system"""
        extracted_path = Path(extracted_data_dir)
        
        # Questions list remains the same as in original script
        questions = [
            {
               
                "renewable_POs": {
                    'availability': 'Read the contract text and answer by your understanding, Does this contract have renewable POs? If yes, extract the details.',
                    'details': [
                        {
                            "name": '',
                            "description": '',
                            "duration": ''
                        }
                    ]
                },
                "multiple_contracts_with_customer": {'yes/no' : 'Read the contract text and answer by your understanding, Are there multiple contracts with the same customer? If so, should they be combined as a single contract under IFRS 15?  - If yes: Provide rationale for combining contracts.',
                "should_they_be_combined_as_one_contract": {
                    "rationale": ''
                }},
                "clarity_of_terms_payment_terms": {
                    "clarity_of_terms": 'Read the contract text and answer by your understanding, Does the contract have clear terms, including payment terms and the goods/services to be transferred?  - If no: List the areas lacking clarity.' ,
                    "areas_lacking_clarity": [
                        {
                            "name": '', 
                            "description": ''
                        }
                    ]
                },
                "performance_obligations": {
                    "distinct_services_promised": 'Read the contract text and answer by your understanding, What distinct services are promised in the contract? Provide a list of services included in the contract.' ,
                    "services_included": [
                        {
                            "name": '',
                            "description": ''
                        }
                    ]
                },
                'services_distinct_in_nature_as_seperate_POs' : {
                    'yes/no' : 'Read the contract text and answer by your understanding, Are the services provided potentially distinct in nature as separate performance obligations . - If yes: Specify how each good/service meets this criterion.',
                    "how_service_meet_criterion": [
                        {
                            "name": '',
                            "description": ''
                        }
                    ]
                },
                "services_distinct_in_terms_of_contract": {
                    'yes/no' : 'Read the contract text and answer by your understanding, Are the services distinct in terms of the contract and identifiable as separate performance obligations?  - Provide details',
                    "details": [
                        {
                            "name": '',
                            "description": ''
                        }
                    ]
                },
                "contract_modifications_or_variable_considerations_affecting_PO": {
                    'yes/no': 'Read the contract text and answer by your understanding, Are there any contract modifications, options, or variable considerations that may affect performance obligations? - If yes: Describe the modifications or options and their potential impact on revenue recognition.  Provide details' ,
                    "modifications_options": [
                        {
                            "name": '',
                            "description": '',
                            "impact_on_revenue": ''
                        }
                    ]
                },
                "transaction_price" : {
                    'pricing_available' : 'Read the contract text and answer by your understanding, What is the total transaction price for the contract? - Provide the total amount applicable on contract date, including any variable consideration.',
                    "total_transaction_price": {
                        "amount_applicable_on_contract_date_including_variable_considerations": ''
                    },
                    "compensation_structure": {
                        'available_yes/no' : 'Read the contract text and answer by your understanding,Is the compensation structure available - Provide details on applicable fee structure: Fixed fee, lump sum, time and materials etc.',
                        "details": {
                            "name": '',
                            "description": ''
                        }
                    },
                    "variable_considerations": {
                        'available_yes/no' : 'Read the contract text and answer by your understanding, Does the contract include variable consideration (e.g., discounts, rebates, performance bonuses, penalties)? - If yes: Describe the nature and how it is estimated.' ,
                        "nature": [
                            {
                                "name": '',
                                "amount": '',
                                "estimation": ''
                            }
                        ]
                    },
                    "financing_components_non_cash_considerations_payable_amt_to_client":  {
                        'yes/no' : 'Read the contract text and answer by your understanding, Are there any significant financing components, non-cash considerations, or payable amounts to clients within the contract? - If yes: Detail how these will be considered in revenue recognition.',
                        "details": [
                            {
                                "name": '',
                                "considerations": ''
                            }
                        ]
                    },
                },
                "other_costs_associated_with_delivery": { 
                    'yes/no' : 'Read the contract text and answer by your understanding, Other costs associated with delivery of these services. Travel expenses, overheads included in a contract fee?  - Provide details',
                    'details' : [
                        {
                            "cost": '',
                            "amount": '',
                            "description": ''
                        }
                    ],},
                    "reimbursable_expenses_billed_to_client_separately": {
                        'available_yes/no' : 'Read the contract text and answer by your understanding, Other project reimbursable expenses which can be billed to the client separately. - If yes: provide details' ,
                        'details' : {
                                "expense": '',
                                "amount":  '',
                                "description": ''
                        }
                    },
                            "non_recoverable_costs": {
                                'yes/no' : 'Read the contract text and answer by your understanding, Other potential non recoverable costs. E.g.Demobilisation - If yes: provide details',
                                'details' : [
                                    {
                                        "name": '',
                                        "amount": '' ,
                                        "description": ''
                                    }
                                ]
                            },

            }]
        
        for contract_dir in extracted_path.iterdir():
            if not contract_dir.is_dir():
                continue
            
            self.logger.info(f"Processing {contract_dir.name}")
            
            contract_output_dir = self.output_dir / contract_dir.name
            contract_output_dir.mkdir(exist_ok=True)
            
            documents = self.load_contract_documents(contract_dir)
            vector_store = self.create_vector_store(documents)
            
            retriever = vector_store.as_retriever(
                search_kwargs={"k": 3}
            )
            
            extracted_data = {}
            
            for section in questions:
                question = json.dumps(section, indent=2)
                retrieved_docs = retriever.invoke(question)
                context = "".join([doc.page_content for doc in retrieved_docs])
                
                response = self.query_model(context, question)
                
                if response:
                    extracted_data.update(response)
                    self.logger.info(f"Added data for question")
                else:
                    self.logger.warning(f"No valid response for question")
            
            if extracted_data:
                output_file = contract_output_dir / f"{contract_dir.name}_final_extracted.json"
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
                    self.logger.info(f"Successfully saved data to {output_file}")
                except Exception as e:
                    self.logger.error(f"Error saving data to {output_file}: {str(e)}")
            else:
                self.logger.warning(f"No data extracted for {contract_dir.name}")


if __name__ == "__main__":
    rag_system = ContractRAGSystem()
    rag_system.process_contracts()

print("Script execution completed.")