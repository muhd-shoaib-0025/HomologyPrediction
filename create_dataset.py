import sys
from io import StringIO
import pandas as pd
import time
import numpy as np
from Bio import Entrez, SeqIO
import threading
from datetime import timedelta
import multiprocessing as mp
min_seq_length = 500
max_seq_length = 2000

def get_protein_name_from_protein_id(protein_id):
    handle = Entrez.efetch(db="protein", id=protein_id, rettype="gb", retmode="text")
    record = handle.read()
    handle.close()
    # Parse the GenBank record to extract the protein name
    protein_name = None
    for line in record.split('\n'):
        if line.startswith('DEFINITION'):
            protein_name = line.split(' ')[1:]
            break

    return ' '.join(protein_name)

Entrez.email="pmartel.at.ualg@gmail.com"
Entrez.api_key="7bd9ad8fc1c3a11cdf4cf53e624b276aa809"
lock = mp.Lock()

def get_protein_info(entrez_gene_id):
    try:
        time.sleep(0.1)
        handle = Entrez.esearch(db="protein", term=str(entrez_gene_id) + '[Gene Id]')
        record = Entrez.read(handle)
        handle.close()
        if record['IdList']:
            time.sleep(0.1)
            protein_id = record['IdList'][0]
            #protein_name = get_protein_name_from_protein_id(protein_id)
            protein_handle = Entrez.efetch(db='protein', id=protein_id, rettype='fasta', retmode='text')
            protein_record = protein_handle.read()
            protein_handle.close()
            protein_name = protein_record.split('\n')[0]
            protein_sequence = ''.join(protein_record.split('\n')[1:])
            return protein_id, protein_name, protein_sequence
    except:
        return ValueError(f"No protein record found for gene symbol: {entrez_gene_id}")

def download_sequence(entrez_gene_id):
    try:
        time.sleep(0.1)
        handle = Entrez.esearch(db="nucleotide", term=str(entrez_gene_id) + '[Gene Id] mrna[filter]')
        record = Entrez.read(handle)
        handle.close()
        # if more than zero nuclotiede sequences were found for the gene id, proceed to fetch the first one
        # NOTE: this assumed that the first sequence is really the one we want.
        # fetch sequences by nucleotide sequence id
        if record['IdList']:
            time.sleep(0.1)
            handle = Entrez.efetch(db="nucleotide", id=record['IdList'][0], rettype="gb", retmode="text")
            seq_record = SeqIO.read(handle, "genbank")
            gene_sequence = str(seq_record.seq)
            handle.close()
            if seq_record.annotations['molecule_type'] == 'mRNA' and len(gene_sequence) >= min_seq_length and len(gene_sequence) <= max_seq_length:  # Check if molecule really is mRNA
                return gene_sequence
                # here you will process seq_record, extracting the relevant information, like
                # seq_record.name, seq_recored.description and seq_record.seq_record
    except:
        raise Exception(f"No gene record found for gene symbol: {entrez_gene_id}")

nucleotides = ['A', 'T', 'C', 'G']

def is_only_nucleotides(input_string):
    return all(nucleotide in nucleotides for nucleotide in input_string)

thread_id_counter = 0
import time

thread_id_counter = 0
MAX_RETRIES = 3  # You can adjust this number as needed

def read_data(df_split, file, num_sequences, families, gene_ids, prot_ids, exceptions, round):
    global thread_id_counter
    thread_id = thread_id_counter
    thread_id_counter += 1
    start_time = time.time()

    for i, row in enumerate(df_split.iterrows()):
        entrez_gene_id = row[1]['entrez_gene_id']
        end_time = time.time()  # Record the end time
        formatted_time = str(timedelta(seconds=end_time - start_time)).split(".")[0]
        print('Reading data ... #rows=' + str(len(df_split)) +
              '; round: ' + str(round) +
              '; i: ' + str(i) +
              '; num_families: ' + str(len(families)) +
              '; num_sequences: ' + str(num_sequences.value) +
              '; exceptions: ' + str(len(exceptions)) +
              '; elapsed time: ' + formatted_time +
              '; thread: ' + str(thread_id))

        retries = 0

        while retries <= MAX_RETRIES:
            try:
                gene_sequence = download_sequence(entrez_gene_id)
                gene_family = row[1]['family_name']
                if entrez_gene_id not in gene_ids and gene_sequence is not None:
                    entrez_protein_id, protein_name, protein_sequence = get_protein_info(entrez_gene_id)
                    if entrez_protein_id not in prot_ids:
                        families.add(gene_family)
                        file.write(f"{entrez_gene_id}\t{gene_family}\t{gene_sequence}\t{entrez_protein_id}\t{protein_name}\t{protein_sequence}\n")
                        num_sequences.value += 1
                        gene_ids.append(entrez_gene_id)
                        prot_ids.append(entrez_protein_id)
                break
            except:
                if retries == MAX_RETRIES:
                    row = row[1].astype(str)
                    exceptions.append(row)
                    break
                retries += 1
                time.sleep(0.1)
                continue

import gc

if __name__ == '__main__':
    gc.collect()
    num_splits = mp.cpu_count()
    i = 0
    with open('gene-families.tsv', "r") as tsv_file:
        tsv_string = tsv_file.read().replace(',', '')
    df = pd.read_csv(StringIO(tsv_string), sep="\t")
    df_splits = np.array_split(df, num_splits)
    num_sequences = mp.Manager().Value('i', 0)
    gene_ids = mp.Manager().list()
    prot_ids = mp.Manager().list()
    exceptions = mp.Manager().list()
    families = set()
    file = open('dataset.csv', 'w')
    file.write(f"entrez_gene_id\tgene_family\tgene_sequence\tentrez_protein_id\tprotein_name\tprotein_sequence\n")
    round = 1
    while True:
        previous_exception_count = len(exceptions)
        threads = list()
        for idx, df_split in enumerate(df_splits):
            df_split = df_split.reset_index(drop=True)
            t = threading.Thread(target=read_data, args=(df_split, file, num_sequences, families, gene_ids, prot_ids, exceptions, round))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
        current_exception_count = len(exceptions)
        round+=1

        exceptions = list(exceptions)
        if current_exception_count != previous_exception_count and current_exception_count > num_splits*2:
            df = pd.concat(exceptions, axis=1).T
            df_splits = np.array_split(df, num_splits)
            exceptions = mp.Manager().list()
        else:
            break

    '''
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_splits) as executor:
        for idx, df_split in enumerate(df_splits):
            df_split = df_split.reset_index(drop=True)
            executor.submit(read_data, df_split, file, num_sequences, families)
        executor.shutdown()
    '''

