"""
Evaluation Script for Lora Fine-Tuned LLMs 
"""

import argparse
import json
import logging
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
BATCH_SIZE = 8
MAX_TOKENS = 512

# Dataset-specific reference keys mapping
DATASET_REFERENCE_KEYS = {
    "amazon_diseases": "Response",
    "wikidoc_pt": "answer",
    "abcfarma_qa": "answer",
    "multiclinsum_pt": "summary",
    "fall_detection": "Target",
    "drbodebench": "resposta",
    "semclinbr": "tags",
    "clinical_ner": "tags",
}


def generate_prompt_drbodebench() -> str:
    """Generate system prompt for DrBodeBench multiple choice questions."""
    base_prompt = (
        "Você é um especialista na área de Medicina, resolvendo uma questão de uma prova de alto nível. "
        "Seu objetivo é demonstrar domínio do assunto, informando na resposta a letra da alternativa correta (dentre A, B, C e D). "
        "Sua resposta deve seguir estritamente o formato:\n\n"
        "Resposta: <LETRA>.\n"
    )
    
    return base_prompt


def generate_prompt_clinical_ner() -> str:
    """Generate system prompt for Clinical NER Custom dataset."""
    entities = {
        "Condição": 'Condição clínica ou diagnóstico. Engloba doenças, sintomas ou condições clínicas. Exemplos: "dor", "febre", "infecção aguda", "otite", "comichão grave".',
        "Anatomia": 'Local anatômico ou partes do corpo humano mencionado no contexto clínico. Exemplos: "orelha esquerda", "região", "cabeça", "abdómen", "região do joelho".',
        "Terapia": 'Terapia, tratamentos, medicamento ou intervenção clínica aplicada. Exemplos: "paracetamol", "antibiótico", "uso de ibuprofeno", "tratamento tópico", "analgésico".',
        "Procedimento": 'Exame médico, teste diagnóstico ou procedimento de imagem. Exemplos: "TC cerebral", "ressonância", "hemograma", "eletrocardiograma", "endoscopia".',
        "Evolução": 'Mudança, egresso ou evolução do quadro clínico ao longo do tempo. Exemplos: "agravamento", "melhora", "persistência", "regressão", "piora".',
        "Genética": 'Fator ou marcador genético associado à condição. Exemplos: "BRCA1", "mutação genética", "gene CFTR", "deleção 22q11", "fator hereditário".',
        "Negação": 'Negação de condição, evento ou sintoma. Exemplos: "sem", "não apresenta dor", "ausência de sinais", "afebril", "sem alterações".',
        "Observação": 'Observações adicionais que não se enquadram em outras categorias. Exemplos: "virar a calça", "realizar ações complexas", "apenas com a mão direita".',
        "Resultado": 'Resultado de exames ou observações objetivas. Exemplos: "normal", "alterado", "positivo", "dentro dos parâmetros".',
        "Tempo": 'Expressões temporais indicando duração ou data de sintomas/eventos. OBS: Idade do paciente não entra aqui. Exemplos: "19 meses", "há três dias", "hoje", "desde ontem".',
        "Valor": 'Valor numérico relevante clinicamente. Medidas numéricas associadas a sinais vitais ou exames (ex.: dosagem, temperatura, pressão). OBS: Idade do paciente não entra aqui. Exemplos: "38ºC", "120/80", "98%", "70 bpm", "5 mg".',
        "ViaAdmin": 'Via de administração de medicamento ou terapia. Exemplos: "via oral", "intravenosa", "tópica", "intramuscular", "subcutânea".',
        "Característica": 'Característica ou descrição que qualifica uma condição clínica. Exemplos: "associado", "diminuição", "semelhante", "recorrente", "tátil".',
    }
    
    entities_texto = "\n".join([f"{k}: {v}" for k, v in entities.items()])
    
    base_prompt = f"""Você é um reconhecedor de entidades nomeadas (NER) especializado em textos clínicos. Sua tarefa é identificar e rotular entidades no texto fornecido.

        As únicas categorias de entidade permitidas são as seguintes: 
        {entities_texto}

        Regras para a resposta:
        - Você deve identificar as entidades no texto e agrupá-las por categoria.
        - A resposta deve ter o seguinte formato:
            "Anatomia: entidade1; entidade2; entidade3
            Condição: entidade1; entidade2
            Resultado: entidade1"
        - Se nenhuma entidade de uma classe estiver presente no texto, não inclua essa classe na resposta.
        - Use os termos exatamente como aparecem no texto, sem corrigir, reescrever, normalizar ou expandir abreviação.
        - Sua resposta deve conter apenas a lista estruturada das entidades. Não inclua explicações ou comentários.
        - OBS: Cada entidade pode ser formada por um ou mais tokens."""
    
    return base_prompt


def generate_prompt_semclinbr() -> str:
    """Generate system prompt for SemClinBr NER dataset."""
    entities = {
        "Disorder": "Doenças, condições clínicas, sinais e sintomas que indicam alterações no estado de saúde. Exemplos: 'sepse', 'dor torácica', 'febre', 'dispneia', 'anemia', 'edema'.",
        "Procedures": "Intervenções e cuidados realizados para diagnóstico, tratamento, prevenção ou acompanhamento. Exemplos: 'angioplastia', 'traqueostomia', 'curativo', 'alta hospitalar', 'fisioterapia respiratória'.",
        "Chemicals and Drugs": "Substâncias químicas, medicamentos e fármacos usados no tratamento, prevenção ou diagnóstico. Exemplos: 'paracetamol', 'adrenalina', 'sinvastatina', 'AAS', 'furosemida'.",
        "Abbreviation": "Siglas e abreviações comuns em registros clínicos. Exemplos: 'PA', 'FC', 'MMII', 'UTI', 'VM', 'HAS'."
    }
    entities_texto = "\n".join([f"{k}: {v}" for k, v in entities.items()])

    base_prompt = f"""Você é um reconhecedor de entidades nomeadas (NER) especializado em textos clínicos. Sua tarefa é identificar e rotular entidades no texto fornecido.

        As únicas categorias de entidade permitidas são as seguintes: 
        {entities_texto}

        Regras para a resposta:
        - Você deve identificar as entidades no texto e agrupá-las por categoria.
        - A resposta deve ter o seguinte formato:
            "Disorder: entidade1; entidade2; entidade3
            Procedures: entidade1; entidade2
            Chemicals and Drugs: entidade1; 
            Abbreviation: entidade1; entidade2; etc;
        - Se nenhuma entidade de uma classe estiver presente no texto, não inclua essa classe na resposta.
        - Use os termos exatamente como aparecem no texto, sem corrigir, reescrever, normalizar ou expandir abreviação.
        - Sua resposta deve conter apenas a lista estruturada das entidades. Não inclua explicações ou comentários.
        - OBS: Cada entidade pode ser formada por um ou mais tokens."""
    
    return base_prompt


def format_prompt(example: Dict, dataset_name: str) -> Tuple[List[Dict], str, Any, str]:
    """
    Generate chat messages, ID, and reference based on dataset type.
    Returns: (messages, original_id, reference, text)
    """
    original_id = example["id"]
    
    reference_key = DATASET_REFERENCE_KEYS.get(dataset_name)
    if not reference_key:
        raise ValueError(f"Reference key not defined for dataset: {dataset_name}")
    reference = example[reference_key]

    if dataset_name == "amazon_diseases":
        system_prompt = (
            "Você é um assistente de saúde virtual. Responda à pergunta a seguir de forma clara, "
            "objetiva e em um único parágrafo de texto corrido. Sua resposta deve ter aproximadamente 65 palavras. "
            "Não utilize listas, tópicos ou bullet points."
        )
        
        question = example["Question"]
        user_prompt = f"Pergunta: {question}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return messages, original_id, reference, question
        
    elif dataset_name == "wikidoc_pt":
        system_prompt = (
           "Você é um especialista em medicina. Sua tarefa é responder à pergunta a seguir de forma detalhada, "
           "informativa e abrangente, mantendo um tom técnico e preciso. "
           "A resposta deve ser apresentada em um parágrafo único e coeso, com aproximadamente 120 palavras. "
           "Não use listas ou marcadores."
        )
        
        question = example["question"]
        user_prompt = f"Pergunta: {question}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return messages, original_id, reference, question

    elif dataset_name == "abcfarma_qa":
        system_prompt = (
            "Você é um assistente especializado em medicamentos. Dada uma pergunta sobre um medicamento, "
            "identifique seu(s) princípio(s) ativo(s). "
            "Instruções de Formatação:"
            "1. Responda exclusivamente com o nome da substância."
            "2. Não inclua nenhuma palavra ou frase adicional. "
            "3. Se houver mais de um princípio ativo, separe-os com um sinal de '+' sem espaços."
        )
        
        question = example["question"]
        user_prompt = f"Pergunta: {question}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return messages, original_id, reference, question

    elif dataset_name == "multiclinsum_pt":
        system_prompt = (
            "Você é um especialista em redação médica. Gere um resumo de um caso clínico destacando diagnóstico"
            "principal, achados clínicos relevantes, contexto do paciente (idade, sexo,antecedentes importantes),"
            "conduta médica e evolução do caso. "
            "O resumo deve ser claro, conciso, técnico e ter 20-25% do comprimento do texto original. "
            "Evite listas, marcadores ou interpretações subjetivas."
        )
        
        text = example["text"]
        user_prompt = f"Caso clínico: {text}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return messages, original_id, reference, text

    elif dataset_name == "fall_detection":
        system_prompt = (
            "Você é um especialista em saúde clínica. Dada a descrição da evolução médica de um paciente, "
            " classifique se ele sofreu queda (classe 1) ou não (classe 0)."
            " Forneça a resposta final seguindo estritamente este formato:\n Queda: <0 ou 1>.\n"
        )
        
        text = example["Evolucao"]
        user_prompt = f"Evolução clínica: {text}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return messages, original_id, str(reference), text
        
    elif dataset_name == "drbodebench":
        system_prompt = generate_prompt_drbodebench()
        enunciado = "Questão: " + example['enunciado']
        alternativas = example["alternativas"]
        descricao_imagem = example.get("img_description", "Nenhuma imagem fornecida.")
        texto_alternativas = "\n".join([f"{letra}) {texto}" for letra, texto in alternativas.items()])
        enunciado = enunciado + f"\nDescrição da Imagem:\n{descricao_imagem}\nAlternativas Apresentadas:\n{texto_alternativas}."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": enunciado}
        ]
        return messages, original_id, str(reference), enunciado
        
    elif dataset_name == "clinical_ner":
        system_prompt = generate_prompt_clinical_ner()
        text = example["text"]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        return messages, original_id, reference, text
        
    elif dataset_name == "semclinbr":
        system_prompt = generate_prompt_semclinbr()
        text = example["text"]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        return messages, original_id, reference, text
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return records


def setup_model_and_tokenizer(model_name: str):
    """Load model and tokenizer."""
    logger.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='left'
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    logger.info("Model and tokenizer loaded successfully")
    return model, tokenizer


def run_batch_inference(
    model,
    tokenizer,
    dataset: List[Dict],
    dataset_name: str,
    output_path: Path,
    batch_size: int = BATCH_SIZE,
    max_tokens: int = MAX_TOKENS,
):
    """Run batch inference on dataset."""
    logger.info(f"Starting inference on {len(dataset)} samples")
    logger.info(f"Batch size: {batch_size}, Max tokens: {max_tokens}")
    
    results = []
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for start_idx in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
            batch_data = dataset[start_idx:start_idx + batch_size]
            
            # Prepare prompts for batch
            prompts_text_batch = []
            batch_metadata = []
            
            for item in batch_data:
                try:
                    messages, item_id, reference, text = format_prompt(item, dataset_name)
                    
                    # Apply chat template
                    try:
                        chat_formatted_prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    except TypeError:
                        # Fallback: use full_prompt
                        full_prompt = "\n".join(
                            f"{m.get('role', '')}: {m.get('content', '')}"
                            for m in messages
                        )
                        chat_formatted_prompt = full_prompt
                    
                    prompts_text_batch.append(chat_formatted_prompt)
                    batch_metadata.append({
                        'id': item_id,
                        'reference': reference,
                        'text': text,
                    })
                    
                except Exception as e:
                    logger.error(f"Error formatting prompt for item {item.get('id')}: {e}")
                    continue
            
            if not prompts_text_batch:
                continue
            
            try:
                # Tokenize batch
                inputs = tokenizer(
                    prompts_text_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(model.device)
                
                # Generate
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False
                )
                
                # Decode only generated tokens
                input_ids_length = inputs.input_ids.shape[1]
                generated_tokens_batch = outputs[:, input_ids_length:]
                generated_texts = tokenizer.batch_decode(
                    generated_tokens_batch,
                    skip_special_tokens=True
                )
                
                # Save results
                for i, generated_text in enumerate(generated_texts):
                    clean_response = generated_text.strip()
                    
                    result = {
                        'id': batch_metadata[i]['id'],
                        'prediction': clean_response,
                        'ground_truth': batch_metadata[i]['reference'],
                        'text': batch_metadata[i]['text'],
                        'task': dataset_name
                    }
                                        
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    results.append(result)
                        
            except Exception as e:
                logger.error(f"Error processing batch starting at {start_idx}: {e}")
                continue
    
    logger.info(f"Inference complete. Results saved to: {output_path}")
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Lora evaluation of LLMs on clinical benchmark'
    )
    
    parser.add_argument(
        '--dataset-path',
        type=Path,
        required=True,
        help='Path to dataset JSONL file (eg test.jsonl)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        required=False,
        help='Path to save predictions (JSONL)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=False,
        help='Model name or path (HuggingFace)'
    )
    
    parser.add_argument(
        '--dataset-name',
        type=str,
        required=True,
        choices=list(DATASET_REFERENCE_KEYS.keys()),
        help='Dataset name for prompt selection'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help=f'Batch size (default: {BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=MAX_TOKENS,
        help=f'Max new tokens (default: {MAX_TOKENS})'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (for testing)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.dataset_path.exists():
        logger.error(f"Dataset path not found: {args.dataset_path}")
        sys.exit(1)

    if args.model is None:
        model_name = 'google/medgemma-4b-it'
    else:
        model_name = args.model

    # Load dataset
    dataset = load_jsonl(args.dataset_path)
    
    if args.max_samples:
        dataset = dataset[:args.max_samples]
        logger.info(f"Limited to {args.max_samples} samples for testing")
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer(model_name)
    
    max_tokens = args.max_tokens

    safe_model_name = model_name.replace("/", "_")
    
    # Output path 
    default_output = (
        Path("../../results/predictions/lora")
        / f"{args.dataset_name}_{safe_model_name}_lora_predictions.jsonl"
    )
    output_path = Path(args.output) if args.output else default_output
    output_path = output_path.resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    try:
        run_batch_inference(
            model,
            tokenizer,
            dataset,
            args.dataset_name,
            output_path,
            args.batch_size,
            max_tokens,
        )
    finally:
        # Cleanup
        logger.info("Cleaning up GPU memory...")
        del model
        torch.cuda.empty_cache()
    
    logger.info("Inference complete!")


if __name__ == "__main__":
    main()