import os
import json
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import concurrent.futures

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict, ValidationError

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Detailed Pydantic Models for Structured Extraction
from typing import List, Dict, Union, Optional
from pydantic import BaseModel, Field, ConfigDict

class KeyContact(BaseModel):
    name: str = Field(default="", description="Name of the key contact person")
    email: str = Field(default="", description="Email address of the contact")
    phone: str = Field(default="", description="Contact phone number")

class BasicInfo(BaseModel):
    project_name: str = Field(default="", description="Name of the project")
    contract_start_date: str = Field(default="", description="Start date of the contract")
    contract_end_date: str = Field(default="", description="End date of the contract")
    total_contract_value: str = Field(default="", description="Total estimated value of the contract")
    brief_description: str = Field(default="", description="Short description of the project")
    key_parties: Dict[str, Union[str, KeyContact]] = Field(
        default_factory=lambda: {
            "customer_name": "",
            "contact_details": KeyContact()
        }
    )

class EnforceableRightsObligation(BaseModel):
    enforceable_rights_obligations: bool = Field(default=False)
    details: Dict[str, str] = Field(
        default_factory=lambda: {
            "contract_date": "",
            "customer_name": "",
            "contract_number": "",
            "contract_type": ""
        }
    )

class ServiceDetail(BaseModel):
    name: str = Field(default="", description="Name of the service or item")
    description: str = Field(default="", description="Detailed description")
    duration: Optional[str] = Field(default="", description="Duration if applicable")

class RenewablePOs(BaseModel):
    availability: bool = Field(default=False)
    details: List[ServiceDetail] = Field(default_factory=list)

class MultipleContracts(BaseModel):
    yes_no: bool = Field(default=False)
    should_they_be_combined_as_one_contract: Dict[str, str] = Field(
        default_factory=lambda: {"rationale": ""}
    )

class ClarityOfTerms(BaseModel):
    clarity_of_terms: bool = Field(default=False)
    areas_lacking_clarity: List[ServiceDetail] = Field(default_factory=list)

class PerformanceObligations(BaseModel):
    distinct_services_promised: bool = Field(default=False)
    services_included: List[ServiceDetail] = Field(default_factory=list)

class ServicesDistinct(BaseModel):
    yes_no: bool = Field(default=False)
    how_service_meet_criterion: List[ServiceDetail] = Field(default_factory=list)

class ModificationOption(BaseModel):
    name: str = Field(default="", description="Name of the modification")
    description: str = Field(default="", description="Description of the modification")
    impact_on_revenue: str = Field(default="", description="Impact on revenue recognition")

class ContractModifications(BaseModel):
    yes_no: bool = Field(default=False)
    modifications_options: List[ModificationOption] = Field(default_factory=list)

class VariableConsideration(BaseModel):
    name: str = Field(default="")
    amount: str = Field(default="")
    estimation: str = Field(default="")

class TransactionPrice(BaseModel):
    pricing_available: bool = Field(default=False)
    total_transaction_price: Dict[str, str] = Field(
        default_factory=lambda: {
            "amount_applicable_on_contract_date_including_variable_considerations": ""
        }
    )
    compensation_structure: Dict[str, Union[bool, ServiceDetail]] = Field(
        default_factory=lambda: {
            "available": False,
            "details": ServiceDetail()
        }
    )
    variable_considerations: Dict[str, Union[bool, List[VariableConsideration]]] = Field(
        default_factory=lambda: {
            "available": False,
            "nature": []
        }
    )
    financing_components: Dict[str, Union[bool, List[ServiceDetail]]] = Field(
        default_factory=lambda: {
            "available": False,
            "details": []
        }
    )

class Cost(BaseModel):
    cost: str = Field(default="")
    amount: str = Field(default="")
    description: str = Field(default="")

class OtherCosts(BaseModel):
    yes_no: bool = Field(default=False)
    details: List[Cost] = Field(default_factory=list)

class ReimbursableExpenses(BaseModel):
    available_yes_no: bool = Field(default=False)
    details: Cost = Field(default_factory=Cost)

class NonRecoverableCosts(BaseModel):
    yes_no: bool = Field(default=False)
    details: List[Cost] = Field(default_factory=list)

class StandalonePrices(BaseModel):
    yes_no: bool = Field(default=False)
    method_used_for_estimation: str = Field(default="")
    description: str = Field(default="")

class PriceAllocation(BaseModel):
    method_used: str = Field(default="")
    description: str = Field(default="")

class DiscountsVariableConsiderations(BaseModel):
    yes_no: bool = Field(default=False)
    impact_on_allocation: str = Field(default="")

class Milestones(BaseModel):
    available_yes_no: bool = Field(default=False)
    details: Dict[str, str] = Field(
        default_factory=lambda: {"how_measured_performance": ""}
    )

class FinancialTerms(BaseModel):
    payment_terms: List[ServiceDetail] = Field(default_factory=list)
    revenue_recognition_overtime_or_pointintime: Dict[str, str] = Field(
        default_factory=lambda: {"basis": ""}
    )
    for_PO_recognition_overtime_method_used_for_measuring_progress: Dict[str, str] = Field(
        default_factory=lambda: {
            "Input_Output_Method": "",
            "basis_of_selecting_the_method": ""
        }
    )
    key_financial_obligations: List[Dict[str, str]] = Field(
        default_factory=lambda: [{
            "payment_withholding": "",
            "payment_withholding_condition": ""
        }]
    )

class ContractAnalysis(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        validate_assignment=True
    )
    
    basic_info: BasicInfo = Field(default_factory=BasicInfo)
    enforceable_rights_obligations: EnforceableRightsObligation = Field(default_factory=EnforceableRightsObligation)
    contract_duration: str = Field(default="")
    renewable_POs: RenewablePOs = Field(default_factory=RenewablePOs)
    multiple_contracts_with_customer: MultipleContracts = Field(default_factory=MultipleContracts)
    clarity_of_terms_payment_terms: ClarityOfTerms = Field(default_factory=ClarityOfTerms)
    performance_obligations: PerformanceObligations = Field(default_factory=PerformanceObligations)
    services_distinct_in_nature_as_seperate_POs: ServicesDistinct = Field(default_factory=ServicesDistinct)
    services_distinct_in_terms_of_contract: ServicesDistinct = Field(default_factory=ServicesDistinct)
    contract_modifications_or_variable_considerations_affecting_PO: ContractModifications = Field(default_factory=ContractModifications)
    transaction_price: TransactionPrice = Field(default_factory=TransactionPrice)
    other_costs_associated_with_delivery: OtherCosts = Field(default_factory=OtherCosts)
    reimbursable_expenses_billed_to_client_separately: ReimbursableExpenses = Field(default_factory=ReimbursableExpenses)
    non_recoverable_costs: NonRecoverableCosts = Field(default_factory=NonRecoverableCosts)
    calculation_of_price_allocation_for_POs: PriceAllocation = Field(default_factory=PriceAllocation)
    standalone_prices_for_each_obligation: StandalonePrices = Field(default_factory=StandalonePrices)
    discounts_variable_considerations_specific_to_obligations: DiscountsVariableConsiderations = Field(default_factory=DiscountsVariableConsiderations)
    milestones_or_kpiS: Milestones = Field(default_factory=Milestones)
    financial_terms: FinancialTerms = Field(default_factory=FinancialTerms)

class ContractRAGSystem:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.2-1B"):
        # Logging setup remains the same
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('contract_rag.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        
        # Load environment variables
        load_dotenv()

        # Get Hugging Face token from environment
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not self.hf_token:
            self.logger.warning("No Hugging Face token found in environment")

        # Output directory
        self.output_dir = Path('final_extracted_data')
        self.output_dir.mkdir(exist_ok=True)

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name='intfloat/e5-large-v2'
        )
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv('CHUNK_SIZE', '4000')),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '200')),
            length_function=len
        )

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name , token=self.hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name , token=self.hf_token)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if os.getenv("USE_GPU", "false").lower() == "true" else -1, 
            max_length=int(os.getenv("MAX_TOKENS", 2048)),
            temperature=float(os.getenv("TEMPERATURE", 0.1)),
            top_p=float(os.getenv("TOP_P", 0.95)),
            repetition_penalty=float(os.getenv("REPEAT_PENALTY", 1.15))
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

    def query_groq(self, context: str, query: str) -> Dict[str, Any]:
        # The original comprehensive prompt template
        prompt_template = """
        You are an advanced contract analysis AI with expertise in financial reporting standards (IFRS 15) with strict adherence to the provided JSON schema.
        If the information is not explicitly stated , read and analyse the contract to interpret and answer the question by your side and try to be data rich. Get more and more relevant data out of the contract. Explain things properly like if you list the services define then properly with context to the contract same with other costs and expenses.

    Your task is to meticulously extract and interpret contract details with:
    - Extreme precision and logical reasoning per schema
    - Reference to field-specific instructions
    - Inference for missing information based on context

    CRITICAL RULES:
    1. Output MUST be valid JSON only - no additional text/markdown
    2. Follow exact schema field names and types
    3. Empty strings "" for missing text
    4. Empty lists [] for missing arrays
    5. False for missing booleans
    6. Include all required fields
    7. No extra fields outside schema
    8. Use true/false for boolean fields

    ANALYTICAL FRAMEWORK:
    1. Exhaustive analysis
    2. Logical inference
    3. Evidence-based interpretation
    4. Hierarchical structure
    5. Cross-referenced sections
    6. Risk/ambiguity identification
    7. JSON-only output
    8. Boolean for yes/no questions
    9. Strict type adherence

    EXTRACTION APPROACH:
    - Extract specified details only
    - Be informative about everything you extract
    - Document partial information gaps
    - Use probabilistic interpretation
    - Provide contextual details
    - List related understanding
    - Maintain objectivity

    CRITICAL THINKING:
    - Explicit statements
    - Logical inferences
    - Risk assessment
    - Section relationships

    Context: {context}
    Query Schema: {query}

    RESPOND WITH VALID JSON ONLY.
        """
        
        try:
            response = self.pipeline(
                prompt_template.format(context=context, query=query),
                max_length=2048,
                num_return_sequences=1,
                truncation=True
            )[0]['generated_text']

            self.logger.info(f"Raw API response content: {response}")
            
            # Clean the response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            
            # Parse the JSON string
            parsed_json = json.loads(response)
            
            # Validate the parsed JSON with Pydantic
            try:
                parsed_content = ContractAnalysis(**parsed_json)
                self.logger.info(f"Successfully parsed and validated JSON")
                return parsed_content.model_dump()
            except ValidationError as e:
                self.logger.error(f"Validation error: {str(e)}")
                self.logger.error(f"Raw content that failed: {content}")
                return {"error": "Failed to validate response", "raw_content": content}
        
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {str(e)}")
            self.logger.error(f"Raw content that failed to decode: {content}")
            return {"error": "Failed to decode JSON", "raw_content": content}
        except Exception as e:
            self.logger.error(f"Error querying Hugging Face model: {str(e)}")
            return {"error": str(e)}

    def merge_responses(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple response dictionaries into a single complete response.
        
        Args:
            responses (List[Dict[str, Any]]): List of response dictionaries to merge
            
        Returns:
            Dict[str, Any]: Merged response dictionary
        """
        merged_response = {}
        
        for response in responses:
            if not response:
                continue
                
            for key, value in response.items():
                if key not in merged_response:
                    merged_response[key] = value
                elif isinstance(value, dict) and isinstance(merged_response[key], dict):
                    # Recursively merge nested dictionaries
                    merged_response[key] = self.merge_responses([merged_response[key], value])
                elif isinstance(value, list) and isinstance(merged_response[key], list):
                    # Combine lists while removing duplicates based on content
                    existing_items = [json.dumps(item, sort_keys=True) for item in merged_response[key]]
                    for new_item in value:
                        if json.dumps(new_item, sort_keys=True) not in existing_items:
                            merged_response[key].append(new_item)
                else:
                    # For primitive types, prefer non-empty values
                    if not merged_response[key] and value:
                        merged_response[key] = value
        
        return merged_response

    def process_contract(self, contract_dir: Path, questions: List[Dict[str, Any]]):
        """Process a single contract using RAG system"""
        self.logger.info(f"Processing {contract_dir.name}")
        
        contract_output_dir = self.output_dir / contract_dir.name
        contract_output_dir.mkdir(exist_ok=True)
        
        documents = self.load_contract_documents(contract_dir)
        vector_store = self.create_vector_store(documents)
        
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 3}
        )
        
        extracted_data = {}
        section_responses = []
        
        for section in questions:
            question = json.dumps(section, indent=2)
            retrieved_docs = retriever.invoke(question)
            context = "".join([doc.page_content for doc in retrieved_docs])
            
            time.sleep(30)  # Wait between questions
            response = self.query_groq(context, question)
            
            if response:
                extracted_data.update(response)
                section_responses.append(response)

        if section_responses:
            merged_data = self.merge_responses(section_responses)
            try:
                validated_data = ContractAnalysis(**merged_data)
                output_data = validated_data.model_dump()
                
                output_file = contract_output_dir / f"{contract_dir.name}_final_extracted.json"
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, indent=2, ensure_ascii=False)
                    self.logger.info(f"Successfully saved merged data to {output_file}")
                except Exception as e:
                    self.logger.error(f"Error saving merged data to {output_file}: {str(e)}")
                    
            except ValidationError as e:
                self.logger.error(f"Validation error for merged data: {str(e)}")
                self.logger.error(f"Raw merged data that failed: {merged_data}")
        else:
            self.logger.warning(f"No responses to merge for {contract_dir.name}")

    def process_contracts(self, extracted_data_dir: str = "extracted_data"):
        """Process all contracts using RAG system"""
        extracted_path = Path(extracted_data_dir)
        
        questions = [
            {
                "basic_info": {
                    "project_name": "",
                    "contract_start_date": "",
                    "contract_end_date": "",
                    "total_contract_value": "",
                    "brief_description": "",
                    "key_parties": {
                        "customer_name": "",
                        "contact_details": {
                            "name": "",
                            "email": "",
                            "phone": ""
                        }
                    }
                },
                "enforceable_rights_obligations": {
                    "enforceable_rights_obligations": "Is there a contract with a customer that creates enforceable rights and obligations? If yes, extract the details.",
                    "details": {
                        "contract_date": "",
                        "customer_name": "",
                        "contract_number": "",
                        "contract_type": ""
                    }
                },
                "contract_duration": "",
                "renewable_POs": {
                    "availability": "Read the contract text and answer by your understanding, Does this contract have renewable POs? If yes, extract the details.",
                    "details": [
                        {
                            "name": "",
                            "description": "",
                            "duration": ""
                        }
                    ]
                },
                "multiple_contracts_with_customer": {
                    "yes/no": "Read the contract text and answer by your understanding, Are there multiple contracts with the same customer? If so, should they be combined as a single contract under IFRS 15?  - If yes: Provide rationale for combining contracts.",
                    "should_they_be_combined_as_one_contract": {
                        "rationale": ""
                    }
                },
                "clarity_of_terms_payment_terms": {
                    "clarity_of_terms": "Read the contract text and answer by your understanding, Does the contract have clear terms, including payment terms and the goods/services to be transferred?  - If no: List the areas lacking clarity.",
                    "areas_lacking_clarity": [
                        {
                            "name": "",
                            "description": ""
                        }
                    ]
                }
            ,
                "performance_obligations": {
                    "distinct_services_promised": "Read the contract text and answer by your understanding, What distinct services are promised in the contract? Provide a list of services included in the contract.",
                    "services_included": [
                        {
                            "name": "",
                            "description": ""
                        }
                    ]
                }},{
                "services_distinct_in_nature_as_seperate_POs": {
                    "yes/no": "Read the contract text and answer by your understanding, Are the services provided potentially distinct in nature as separate performance obligations . - If yes: Specify how each good/service meets this criterion.",
                    "how_service_meet_criterion": [
                        {
                            "name": "",
                            "description": ""
                        }
                    ]
                },
                "services_distinct_in_terms_of_contract": {
                    "yes/no": "Read the contract text and answer by your understanding, Are the services distinct in terms of the contract and identifiable as separate performance obligations?  - Provide details",
                    "details": [
                        {
                            "name": "",
                            "description": ""
                        }
                    ]
                },
                "contract_modifications_or_variable_considerations_affecting_PO": {
                    "yes/no": "Read the contract text and answer by your understanding, Are there any contract modifications, options, or variable considerations that may affect performance obligations? - If yes: Describe the modifications or options and their potential impact on revenue recognition.  Provide details",
                    "modifications_options": [
                        {
                            "name": "",
                            "description": "",
                            "impact_on_revenue": ""
                        }
                    ]
                },
                "transaction_price": {
                    "pricing_available": "Read the contract text and answer by your understanding, What is the total transaction price for the contract? - Provide the total amount applicable on contract date, including any variable consideration.",
                    "total_transaction_price": {
                        "amount_applicable_on_contract_date_including_variable_considerations": ""
                    },
                    "compensation_structure": {
                        "available_yes/no": "Read the contract text and answer by your understanding,Is the compensation structure available - Provide details on applicable fee structure: Fixed fee, lump sum, time and materials etc.",
                        "details": {
                            "name": "",
                            "description": ""
                        }
                    },
                    "variable_considerations": {
                        "available_yes/no": "Read the contract text and answer by your understanding, Does the contract include variable consideration (e.g., discounts, rebates, performance bonuses, penalties)? - If yes: Describe the nature and how it is estimated.",
                        "nature": [
                            {
                                "name": "",
                                "amount": "",
                                "estimation": ""
                            }
                        ]
                    },
                    "financing_components_non_cash_considerations_payable_amt_to_client": {
                        "yes/no": "Read the contract text and answer by your understanding, Are there any significant financing components, non-cash considerations, or payable amounts to clients within the contract? - If yes: Detail how these will be considered in revenue recognition.",
                        "details": [
                            {
                                "name": "",
                                "considerations": ""
                            }
                        ]
                    }
                }
            },
                {
                "other_costs_associated_with_delivery": {
                    "yes/no": "Read the contract text and answer by your understanding, Other costs associated with delivery of these services. Travel expenses, overheads included in a contract fee?  - Provide details",
                    "details": [
                        {
                            "cost": "",
                            "amount": "",
                            "description": ""
                        }
                    ]
                },
                "reimbursable_expenses_billed_to_client_separately": {
                    "available_yes/no": "Read the contract text and answer by your understanding, Other project reimbursable expenses which can be billed to the client separately. - If yes: provide details",
                    "details": {
                        "expense": "",
                        "amount": "",
                        "description": ""
                    }
                },
                "non_recoverable_costs": {
                    "yes/no": "Read the contract text and answer by your understanding, Other potential non recoverable costs. E.g.Demobilisation - If yes: provide details",
                    "details": [
                        {
                            "name": "",
                            "amount": "",
                            "description": ""
                        }
                    ]
                },
                "calculation_of_price_allocation_for_POs": {
                    "method_used": "Read the contract text and answer by your understanding, How is the transaction price allocated among the performance obligations? - Describe the method used (e.g., relative standalone selling prices).",
                    "description": "Read the contract text and answer by your understanding, Describe the method used."
                },
                "standalone_prices_for_each_obligation": {
                    "yes/no": "Read the contract text and answer by your understanding, Are standalone selling prices available for each performance obligation? - If no: Describe the estimation method used.",
                    "method_used_for_estimation": "",
                    "description": ""
                },
                "discounts_variable_considerations_specific_to_obligations": {
                    "yes/no": "Read the contract text and answer by your understanding, Does any discount or variable consideration apply specifically to one or more performance obligations?  - If yes: Specify how it impacts allocation.",
                    "impact_on_allocation": ""
                },
                "milestones_or_kpiS": {
                    "available_yes/no": "Read the contract text and answer by your understanding, Are there milestones or KPIs applicable to the contract?  - provide details how performance in measured",
                    "details": {
                        "how_measured_performance": ""
                    }
                },
                "financial_terms": {
                    "payment_terms": [
                        {
                            "name": "",
                            "description": ""
                        }
                    ],
                    "revenue_recognition_overtime_or_pointintime": {
                        "basis": "Read the contract text and answer by your understanding, Read the contract text and answer if the revenue is recognized over time or at a point in time. If overtime, provide the basis (input/output method)."
                    },
                    "for_PO_recognition_overtime_method_used_for_measuring_progress": {
                        "Input/Output Method": "Read the contract text and answer by your understanding, For performance obligations recognized over time, what method is used to measure progress? (Input/Output method)",
                        "basis_of_selecting_the_method": "Read the contract text and answer by your understanding, and for performance obligations recognized over time, what method is used to measure progress? Provide the basis of selecting the model."
                    },
                    "key_financial_obligations": [
                        {
                            "payment_withholding": "",
                            "payment_withholding_condition": ""
                        }
                    ]
                }
            }
        ]
        
        # Use ThreadPoolExecutor to process contracts in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(self.process_contract, contract_dir, questions) for contract_dir in extracted_path.iterdir() if contract_dir.is_dir()]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Error processing contract: {str(e)}")

if __name__ == "__main__":
    rag_system = ContractRAGSystem()
    rag_system.process_contracts()

print("Script execution completed.")