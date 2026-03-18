"""
LoRA Fine-tuning for Clinical Benchmark PT
Uses the SAME PROMPTS as Zero-Shot inference.
Supports: QA, multiple_choice, Summarization, Classification, NER
"""

import argparse
import torch
import json
from pathlib import Path
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from typing import Dict, List, Tuple, Any

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

# ========================================================
# Dataset reference keys
# ========================================================

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
# ========================================================
# Helper prompt generators
# ========================================================


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

# ========================================================
# Unified PROMPT BUILDER
# ========================================================

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
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": example["Response"]}
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
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": example["answer"]},
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
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": example["answer"]},
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
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": example["summary"]}
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
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "Queda: "+str(example["Target"])+"."}
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
            {"role": "user", "content": enunciado},
            {"role": "assistant", "content": "Resposta: "+str(example["resposta"])+'.'}
        ]
        return messages, original_id, str(reference), enunciado
        
    elif dataset_name == "clinical_ner":
        system_prompt = generate_prompt_clinical_ner()
        text = example["text"]
        tags_dict = example['tags']
        tags_lines = []
        for category, entities in tags_dict.items():
            if entities:  
                entities_str = "; ".join(entities)
                tags_lines.append(f"{category}: {entities_str}")
        tags_str = "\n".join(tags_lines)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
            {"role": "assistant", "content": tags_str},
        ]
        return messages, original_id, reference, text
        
    elif dataset_name == "semclinbr":
        system_prompt = generate_prompt_semclinbr()
        text = example["text"]
        tags_dict = example['tags']
        tags_lines = []
        for category, entities in tags_dict.items():
            if entities:  
                entities_str = "; ".join(entities)
                tags_lines.append(f"{category}: {entities_str}")
        tags_str = "\n".join(tags_lines)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
            {"role": "assistant", "content": tags_str},
        ]
        return messages, original_id, reference, text
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# ============================================================

def format_prompt_for_lora(example, dataset_name):
    """
    Wrap the zero-shot prompt builder so it returns {"messages": ...}
    """

    # Reuse the EXACT same zero-shot formatter
    messages, _id, reference, text = format_prompt(example, dataset_name)

    return {"messages": messages}


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_lora(
    model_name: str,
    dataset_name: str,
    data_dir: Path,
    output_dir: Path,
    epochs: int = 3
):
    logger.info(f"🚀 Starting LoRA training for {dataset_name}")

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---------------------------------------------------------
    # BitsAndBytes 4bit quantization
    # ---------------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(model)

    # ---------------------------------------------------------
    # LoRA config
    # ---------------------------------------------------------
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)
    model.train()

    # ---------------------------------------------------------
    # Load dataset jsonl
    # ---------------------------------------------------------
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "dev.jsonl"),
        },
    )

    train_ds = dataset["train"].map(
        lambda ex: format_prompt_for_lora(ex, dataset_name)
    )

    val_ds = dataset["validation"].map(
        lambda ex: format_prompt_for_lora(ex, dataset_name)
    )

    # ---------------------------------------------------------
    # SFT Training configuration
    # ---------------------------------------------------------
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        logging_steps=10,
        learning_rate=2e-4,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        save_strategy="no",
        eval_strategy="epoch",
        report_to="none",
        dataset_text_field="messages",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        args=training_args,
    )

    trainer.train()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    logger.info("💾 Saving model and tokenizer...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    logger.info("🎉 Training completed successfully!")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", type=str, default="google/medgemma-4b-it")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("../../models/lora"))
    parser.add_argument("--epochs", type=int, default=3)

    args = parser.parse_args()
    model_tag = args.model_name.replace("/", "_")

    outdir = args.output_dir /  args.dataset_name / model_tag
    outdir.mkdir(parents=True, exist_ok=True)

    train_lora(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        output_dir=outdir,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
