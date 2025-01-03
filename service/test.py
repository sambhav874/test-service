import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document

# Pydantic Models
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
    pricing_available: str = Field(default="")
    total_transaction_price: Dict[str, str] = Field(
        default_factory=lambda: {
"amount_applicable_on_contract_date_including_variable_considerations": ""
        }
    )
    compensation_structure: Dict[str, Union[str, CompensationStructure]] = Field(
        default_factory=lambda: {
            "available_yes/no": "",
            "details": CompensationStructure()
        }
    )
    variable_considerations: Dict[str, Union[str, List[VariableConsideration]]] = Field(
        default_factory=lambda: {
            "available_yes/no": "",
            "nature": []
        }
    )
    financing_components_non_cash_considerations_payable_amt_to_client: Dict[str, Union[str, List[FinancingComponent]]] = Field(
        default_factory=lambda: {
            "yes/no": "",
            "details": []
        }
    )

class ContractAnalysisModel(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        validate_assignment=True
    )

    renewable_POs: Dict[str, Union[str, List[RenewablePODetail]]] = Field(
        default_factory=lambda: {
            "availability": "",
            "details": []
        }
    )

    multiple_contracts_with_customer: Dict[str, Union[str, Dict[str, str]]] = Field(
        default_factory=lambda: {
            "yes/no": "",
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

    services_distinct_in_nature_as_seperate_POs: Dict[str, Union[str, List[ServiceDetail]]] = Field(
        default_factory=lambda: {
            "yes/no": "",
            "how_service_meet_criterion": []
        }
    )

    services_distinct_in_terms_of_contract: Dict[str, Union[str, List[ServiceDetail]]] = Field(
        default_factory=lambda: {
            "yes/no": "",
            "details": []
        }
    )

    contract_modifications_or_variable_considerations_affecting_PO: Dict[str, Union[str, List[ModificationOption]]] = Field(
        default_factory=lambda: {
            "yes/no": "",
            "modifications_options": []
        }
    )

    transaction_price: TransactionPriceDetails = Field(default_factory=TransactionPriceDetails)

    other_costs_associated_with_delivery: Dict[str, Union[str, List[AdditionalCost]]] = Field(
        default_factory=lambda: {
            "yes/no": "",
            "details": []
        }
    )

    reimbursable_expenses_billed_to_client_separately: Dict[str, Union[str, ReimbursableExpense]] = Field(
        default_factory=lambda: {
            "available_yes/no": "",
            "details": ReimbursableExpense()
        }
    )

    non_recoverable_costs: Dict[str, Union[str, List[NonRecoverableCost]]] = Field(
        default_factory=lambda: {
            "yes/no": "",
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

        # Initialize model configurations
        self.model_name = model_name or os.getenv('MODEL_NAME', 'meta-llama/Llama-3.2-1B')
        self.hf_token = os.getenv('HUGGINGFACE_API_KEY', 'your_token_here')

        if not self.hf_token:
            self.logger.warning("No Hugging Face token found in environment")

        # Initialize tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token
                , trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.hf_token
                , trust_remote_code=True
            )

            # Create text generation pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                temperature=float(os.getenv('MODEL_TEMPERATURE', '0.1')),
                max_new_tokens=int(os.getenv('MAX_NEW_TOKENS', '500'))
            )
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            raise

        # Initialize output directory
        self.output_dir = Path(os.getenv('FINAL_OUTPUT_DIR', 'final_extracted_data'))
        self.output_dir.mkdir(exist_ok=True)

        # Initialize embeddings
        try:
            embedding_model = os.getenv('EMBEDDING_MODEL', 'intfloat/e5-large-v2')
            os.environ["HUGGINGFACE_API_KEY"] = self.hf_token
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model , model_kwargs={"trust_remote_code": True}
            )
        except Exception as e:
            self.logger.error(f"Error initializing embeddings: {str(e)}")
            raise

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv('CHUNK_SIZE', '4000')),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '200')),
            length_function=len
        )

        # Load questions schema
        self.questions = self.load_questions_schema()

    def load_questions_schema(self) -> Dict:
        """Load the questions schema"""
        return {
                "questions": [
              {
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
                  },
                  "performance_obligations": {
                      "distinct_services_promised": "Read the contract text and answer by your understanding, What distinct services are promised in the contract? Provide a list of services included in the contract.",
                      "services_included": [
                          {
                              "name": "",
                              "description": ""
                          }
                      ]
                  },
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
                  },
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
                  }
              }
          ]

        }

    def load_contract_documents(self, contract_dir: str = "extracted_data") -> List[Document]:
        """Load contract documents directly from extracted_data folder"""
        markdown_path = Path(contract_dir) / 'contract.md'

        try:
            with open(markdown_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.logger.info(f"Successfully loaded contract from {markdown_path}")
        except Exception as e:
            self.logger.error(f"Error reading markdown: {e}")
            content = ""

        full_text = f"Contract Content:\n{content}"

        return [Document(
            page_content=full_text,
            metadata={"source": str(markdown_path)}
        )]

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create FAISS vector store from documents"""
        try:
            texts = self.text_splitter.split_documents(documents)
            vector_store = FAISS.from_documents(texts, self.embeddings)
            self.logger.info("Successfully created vector store")
            return vector_store
        except Exception as e:
            self.logger.error(f"Error creating vector store: {str(e)}")
            raise


    def query_model(self, context: str, query: str) -> Dict[str, Any]:
        """Query the model with context and specific question"""
        prompt_template = """
        You are an advanced contract analysis AI specialized in IFRS 15 analysis.

        Your task is to analyze the contract details and respond ONLY with a JSON object.

        Rules:
        1. Return ONLY the JSON object - no additional text
        2. Follow the exact structure provided in the query
        3. Do not add any explanations or comments
        4. Ensure all JSON keys match exactly

        Context: {context}

        Expected Response Structure: {query}

        Respond with ONLY the JSON object:
        """

        try:
            full_prompt = prompt_template.format(context=context, query=query)
            response = self.pipe(
                full_prompt,
                do_sample=False,  # Disable sampling for more deterministic output
                temperature=0.1,  # Keep temperature very low
                max_new_tokens=1000,
                repetition_penalty=1.0
            )[0]['generated_text']

            # Find JSON content
            try:
                # Remove the original prompt from the response
                response = response[len(full_prompt):]

                # Find the first '{' and last '}'
                json_start = response.find('{')
                if json_start == -1:
                    raise ValueError("No JSON content found")

                json_content = response[json_start:]
                json_end = json_content.rfind('}') + 1
                if json_end == 0:
                    raise ValueError("No closing brace found")

                json_content = json_content[:json_end]

                # Clean the JSON string
                json_content = self._clean_json_string(json_content)

                # Parse and validate JSON
                parsed_json = json.loads(json_content)

                # If we're processing a section, wrap it in the expected structure
                query_dict = json.loads(query)
                section_name = list(query_dict.keys())[0]

                # Ensure the response matches the expected structure
                if section_name not in parsed_json:
                    parsed_json = {section_name: parsed_json}

                # Validate against model
                section_model = ContractAnalysisModel(**{section_name: parsed_json[section_name]})
                return {section_name: getattr(section_model, section_name)}

            except json.JSONDecodeError as je:
                self.logger.error(f"JSON decode error: {str(je)}")
                self.logger.debug(f"Problematic JSON content: {json_content}")
                return {"error": "Invalid JSON format", "details": str(je)}

        except Exception as e:
            self.logger.error(f"Error in query_model: {str(e)}")
            return {"error": str(e)}

    def _clean_json_string(self, json_str: str) -> str:
        """Clean and normalize JSON string"""
        # Remove any leading/trailing whitespace
        json_str = json_str.strip()

        # Remove any text before the first {
        start_idx = json_str.find('{')
        if start_idx != -1:
            json_str = json_str[start_idx:]

        # Remove any text after the last }
        end_idx = json_str.rfind('}')
        if end_idx != -1:
            json_str = json_str[:end_idx + 1]

        # Replace any invalid escape sequences
        json_str = json_str.replace('\n', ' ').replace('\r', ' ')

        # Fix common JSON formatting issues
        json_str = re.sub(r'(?<!\\)"(\w+)":', r'"\1":', json_str)  # Fix unquoted keys
        json_str = re.sub(r'\'', '"', json_str)  # Replace single quotes with double quotes
        json_str = re.sub(r'None', 'null', json_str)  # Replace Python None with JSON null

        return json_str

    def process_contracts(self):
        """Process contracts and extract information"""
        self.logger.info("Starting contract processing")

        try:
            documents = self.load_contract_documents()
            if not documents:
                self.logger.error("No documents loaded")
                return

            vector_store = self.create_vector_store(documents)
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})

            # Initialize results with empty structure
            extracted_data = ContractAnalysisModel().model_dump()

            # Process each section with retries
            for question in self.questions["questions"]:
                for section, section_data in question.items():
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            # Create focused query for this section
                            section_query = {section: section_data}
                            query_str = json.dumps(section_query, indent=2)

                            # Get context
                            retrieved_docs = retriever.invoke(query_str)
                            context = "\n".join([doc.page_content for doc in retrieved_docs])

                            # Query model
                            response = self.query_model(context, query_str)

                            if response and "error" not in response:
                                extracted_data[section] = response[section]
                                self.logger.info(f"Successfully processed section: {section}")
                                break
                            else:
                                self.logger.warning(f"Attempt {attempt + 1} failed for section {section}: {response.get('error', 'Unknown error')}")
                                if attempt == max_retries - 1:
                                    self.logger.error(f"Failed to process section {section} after {max_retries} attempts")

                        except Exception as e:
                            self.logger.error(f"Error processing section {section}, attempt {attempt + 1}: {str(e)}")
                            if attempt == max_retries - 1:
                                raise

            # Save results
            if any(extracted_data.values()):
                output_file = self.output_dir / "final_extracted.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(extracted_data, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Successfully saved data to {output_file}")
            else:
                self.logger.warning("No data extracted")

        except Exception as e:
            self.logger.error(f"Error in process_contracts: {str(e)}")
            raise
def main():
    try:
        rag_system = ContractRAGSystem()
        rag_system.process_contracts()
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()