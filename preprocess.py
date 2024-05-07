import numpy as np
import pandas as pd
import os
import obonet
from Bio import SeqIO, SeqRecord
from pathlib import Path

N_EXTRACTED_TERMS = {"BPO": 1500, "CCO": 800, "MFO": 800}


def load_train_data(train_dir: Path):
    """Read in train data"""
    train_sequences_fasta = list(SeqIO.parse(os.path.join(train_dir, "train_sequences.fasta"), "fasta"))
    train_taxonomy = pd.read_csv(os.path.join(train_dir, "train_taxonomy.tsv"), sep="\t")
    train_terms = pd.read_csv(os.path.join(train_dir, "train_terms.tsv"), sep="\t")

    with open(os.path.join(train_dir, "go-basic.obo")) as obo_file:
        go_graph = obonet.read_obo(obo_file)

    information_accretion = pd.read_csv("IA.txt", sep="\t", header=None, names=["GO_term", "IA"])

    return train_sequences_fasta, train_taxonomy, train_terms, go_graph, information_accretion

def load_test_data(test_dir: Path):
    """Read in test data"""
    test_sequences_fasta = list(SeqIO.parse(os.path.join(test_dir, "testsuperset.fasta"), "fasta"))
    test_taxonomy = pd.read_csv(os.path.join(test_dir, "testsuperset-taxon-list.tsv"), sep="\t", encoding="ISO-8859-1")
    return test_sequences_fasta, test_taxonomy


def create_sequence_dataframe_from_fasta(fasta_sequences: list[SeqRecord.SeqRecord]) -> pd.DataFrame:
    """Create DataFrame for train/test sequences"""
    
    seq_record_dict = {
        "EntryID": [],
        "description": [],
        "sequence": [],
    }
    for seq_record in fasta_sequences:
        seq_record_dict["EntryID"].append(seq_record.id)
        seq_record_dict["description"].append(seq_record.description)
        seq_record_dict["sequence"].append(str(seq_record.seq))

    return pd.DataFrame(seq_record_dict).drop_duplicates()

def augment_train_features_from_fasta_description(sequence_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Extract features from the fasta header text"""
    sequence_dataframe = sequence_dataframe.copy()

    # Extract information from fasta headers
    description_parts = sequence_dataframe.description.str.split(r"\|", n=2, expand=True)
    extracted_groups = sequence_dataframe.description.str.extract(r"([A-Z0-9]+)_([A-Z0-9]+) (.*?) OS=")
    
    # Create features
    sequence_dataframe = sequence_dataframe.assign(
        db=description_parts[0].str[-2:],
        description=description_parts[2],
        
        entry_name_prefix=extracted_groups[0],
        entry_name_suffix=extracted_groups[1],
        protein_name=extracted_groups[2],
        
        organism_name=sequence_dataframe.description.str.extract(r"OS=(.+) OX="),            # The source organism's name
        organism_id=sequence_dataframe.description.str.extract(r"OX=(.+?) ").astype(float),  # The source organism's id
        gene_name=sequence_dataframe.description.str.extract(r"GN=(.+?) "),                  # Optional: Might be empty
        protein_existence=sequence_dataframe.description.str.extract(r"PE=(.+?) ").astype(float),
        sequence_version=sequence_dataframe.description.str.extract(r"SV=(.+?)$").astype(float),
    )

    # Drop unneeded column
    return sequence_dataframe.drop(columns="description")


def augment_test_features_from_fasta_description(sequence_dataframe: pd.DataFrame) -> pd.DataFrame:
    return sequence_dataframe.assign(ID=sequence_dataframe.description.str.split("\t").str[1].astype(int)).drop(columns="description")


def load_annotation_dates():
    return (
        pd.read_csv("Annotated_dates_list.csv")
        .rename(columns=lambda x: x.lower())
        .pipe(lambda df: 
            df.assign(
                first_release=pd.to_datetime(df.first_release)
            )
        )
    )

def get_train_sequences(train_dir: Path):
    train_sequences_fasta, train_taxonomy, _, _, _ = load_train_data(train_dir)
    train_sequences = create_sequence_dataframe_from_fasta(train_sequences_fasta)
    train_sequences = augment_train_features_from_fasta_description(train_sequences)

    return (
        train_sequences
        .merge(train_taxonomy, on="EntryID", how="left")
        .drop_duplicates()
        .merge(load_annotation_dates(), left_on="EntryID", right_on="index", how="left")
        .drop(columns="index")
        # Unfortunately, we have to discard most of the features because the test data does not have those information in its fasta header
        [["EntryID", "sequence", "organism_id", "taxonomyID", "first_release"]]
    )


########### Load PROTGOAT Embeddings ###########

def load_Y_data(ontology: str, embedding_dir: Path, verbose: bool = False):

    if ontology not in N_EXTRACTED_TERMS.keys():
        print('Error: ontology not recognized')
        return None, None, None
    
    # Set up file paths
    onto_abbr = ontology[:2]
    n_extracted_terms = N_EXTRACTED_TERMS[ontology]

    Y_filepath = os.path.join(embedding_dir, f'Y_{onto_abbr}_{n_extracted_terms}.npy')
    Y_labels_filepath = os.path.join(embedding_dir, f'Y_{onto_abbr}_labels_{n_extracted_terms}.npy')
    df_filepath = os.path.join(embedding_dir, f'{ontology}_{n_extracted_terms}_freq_weights.csv')

    # Load data
    Y = np.load(Y_filepath, allow_pickle=True)
    Y_labels = np.load(Y_labels_filepath, allow_pickle=True)
    df = pd.read_csv(df_filepath, index_col=0)
    weights_raw = df['IA_weight'].values.tolist()
    weights = {i: weights_raw[i] for i in range(len(weights_raw))}
    
    if verbose:
        print(f'Loaded {ontology} ontology')

    return Y, Y_labels, weights, df

def load_t5_data(embedding_dir: Path, verbose: bool = False):
    train_data1 = np.load(os.path.join(embedding_dir, 't5_train_data_sorted_f32.npy'), mmap_mode='r')
    test_data1 = np.load(os.path.join(embedding_dir, 't5_test_data_sorted_f32.npy'), mmap_mode='r')
    if verbose:
        print(f'T5 train data shape: {train_data1.shape}')
        print(f'T5 test data shape: {test_data1.shape}')

    return train_data1, test_data1

def load_esm2_s_data(embedding_dir: Path, verbose: bool = False):
    train_data2 = np.load(os.path.join(embedding_dir, 'esm2_train_data_sorted_f32.npy'), mmap_mode='r')
    test_data2 = np.load(os.path.join(embedding_dir, 'esm2_test_data_sorted_f32.npy'), mmap_mode='r')
    if verbose:
        print(f'ESM2 small train data shape: {train_data2.shape}')
        print(f'ESM2 small test data shape: {test_data2.shape}')

    return train_data2, test_data2

def load_esm2_l_data(embedding_dir: Path, verbose: bool = False):
    train_data3 = np.load(os.path.join(embedding_dir, 'ESM2_3B_train_embeddings_sorted.npy'), mmap_mode='r')
    test_data3 = np.load(os.path.join(embedding_dir, 'ESM2_3B_test_embeddings_sorted.npy'), mmap_mode='r')
    if verbose:
        print(f'ESM2 3B train data shape: {train_data3.shape}')
        print(f'ESM2 3B data shape: {test_data3.shape}')

    return train_data3, test_data3

def load_pb_data(embedding_dir: Path, verbose: bool = False):
    train_data4 = np.load(os.path.join(embedding_dir, 'pb_train_data_sorted_f32.npy'), mmap_mode='r')
    test_data4 = np.load(os.path.join(embedding_dir, 'pb_test_data_sorted_f32.npy'), mmap_mode='r')
    if verbose:
        print(f'PB train data shape: {train_data4.shape}')
        print(f'PB data shape: {test_data4.shape}')

    return train_data4, test_data4

def load_ankh_data(embedding_dir: Path, verbose: bool = False):
    train_data5 = np.load(os.path.join(embedding_dir, 'Ankh_train_embeddings_sorted.npy'), mmap_mode='r')
    test_data5 = np.load(os.path.join(embedding_dir, 'Ankh_test_embeddings_sorted.npy'), mmap_mode='r')
    if verbose:
        print(f'Ankh train data shape: {train_data5.shape}')
        print(f'Ankh data shape: {test_data5.shape}')

    return train_data5, test_data5

def load_taxa_data(embedding_dir: Path, verbose: bool = False):
    train_data6 = np.load(os.path.join(embedding_dir, 'protein_taxa_matrix_train.npy'), mmap_mode='r')
    test_data6 = np.load(os.path.join(embedding_dir, 'protein_taxa_matrix_test.npy'), mmap_mode='r')
    
    train_data6 = np.expand_dims(train_data6, axis=1)
    test_data6 = np.expand_dims(test_data6, axis=1)
    if verbose:
        print(f'Taxa train data shape: {train_data6.shape}')
        print(f'Taxa data shape: {test_data6.shape}')

    return train_data6, test_data6

def load_text_embed(embedding_dir: Path, verbose: bool = False):
    
    train_data7 = np.load(os.path.join(embedding_dir, 'Text_abstract_embeds_train_sorted.npy'), mmap_mode='r')
    test_data7 = np.load(os.path.join(embedding_dir, 'Text_abstract_embeds_test_sorted.npy'), mmap_mode='r')
    if verbose:
        print(f'Text abstract data shape: {train_data7.shape}')
        print(f'Text abstract shape: {test_data7.shape}')

    return train_data7, test_data7
