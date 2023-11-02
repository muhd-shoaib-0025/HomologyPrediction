def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import math
import random
import subprocess
from Bio.Blast import NCBIXML
import platform
import numpy as np
import tensorflow as tf
from keras.engine.base_layer_v1 import Layer
from keras.models import Model
from keras.layers import Input, Dense, concatenate, Dropout, Embedding, LayerNormalization, MultiHeadAttention, GlobalAveragePooling2D
from keras import Sequential
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import create_dataset
import util
np.random.seed(42)
tf.random.set_seed(42)

min_seq_per_fam = 5
max_seq_per_fam = 50

min_seq_length = 1000
max_seq_length = 2000

import tensorflow as tf

import os

# Suppress TensorFlow info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 (default) to 3 (suppress everything)

nucleotides = ['A', 'T', 'C', 'G']
import argparse
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"),
             Dense(embed_dim), ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(Layer):
    def __init__(self, max_len, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=max_len, output_dim=embed_dim)

    def call(self, x):
        max_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=max_len, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

import gc

if __name__ == '__main__':
    gc.collect()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--model', type=str, help='Specify the model')
    parser.add_argument('-sample', '--sample_size', type=float, help='Sample Size')
    parser.add_argument('-ff_dim', '--ff_dim', type=int, default=32, help='ff_dim')
    parser.add_argument('-num_heads', '--num_heads', type=int, default=1, help='num_heads')
    parser.add_argument('-hid', '--hidden_layer_size', default=64, type=int, help='Hidden layer size')
    parser.add_argument('-e', '--num_epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('-acc', '--accumulation_steps', type=int, default=4, help='Accumulation steps')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate')

    args = parser.parse_args()

    print(args)

    # Set the provided arguments
    model = args.model
    sample_size = args.sample_size
    ff_dim = args.ff_dim
    num_heads = args.num_heads
    hidden_layer_size = args.hidden_layer_size
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    accumulation_steps = args.accumulation_steps
    learning_rate = args.learning_rate

    random.seed(42)
    gene_data = dict()
    prot_data = dict()
    with open('dataset.csv', 'r') as file:
        file.__next__()
        lines = file.readlines()
    lines = random.sample(lines, int(len(lines)*sample_size))
    for i, line in enumerate(lines):
        gene_id, gene_family, gene_sequence, entrez_protein_id, protein_name, protein_sequence = line.strip().split('\t')
        if len(gene_sequence) >= min_seq_length and len(gene_sequence) <= max_seq_length and create_dataset.is_only_nucleotides(gene_sequence):
            if gene_family not in gene_data and gene_family not in prot_data:
                gene_data[gene_family] = list()
                prot_data[gene_family] = list()
            gene_sequences = gene_data[gene_family]
            prot_sequences = prot_data[gene_family]
            gene_sequences.append((gene_id, gene_sequence))
            prot_sequences.append(f"{protein_name}|{gene_id}\n{protein_sequence}\n")
            gene_data[gene_family] = gene_sequences
            prot_data[gene_family] = prot_sequences

    gene_data = {key: random.sample(value, min(len(value), max_seq_per_fam)) for key, value in gene_data.items()}
    gene_data = {key: value for key, value in gene_data.items() if len(value) >= min_seq_per_fam}

    prot_data = {key: random.sample(value, min(len(value), max_seq_per_fam)) for key, value in prot_data.items()}
    prot_data = {key: value for key, value in prot_data.items() if len(value) >= min_seq_per_fam}

    file = open('blastp/sequences.fasta', 'w')
    for key, value in prot_data.items():
        for v in value:
            file.write(v)
    file.close()
    makeblastdb_cmd = "makeblastdb -in blastp/sequences.fasta -parse_seqids -dbtype prot -out blastp/database"
    blast_cmd = "psiblast -db blastp/database -query blastp/sequences.fasta -out blastp/results.xml -outfmt 5 -max_target_seqs 50"
    if platform.system() == 'Windows':
        os.system('cmd /c '+makeblastdb_cmd)
        os.system('cmd /c '+blast_cmd)
    else:
        subprocess.run(f"{makeblastdb_cmd}", shell=True)
        subprocess.run(f"{blast_cmd}", shell=True)
    blast_results = dict()
    E_VALUE_THRESH = 0.1
    for record in tqdm(NCBIXML.parse(open("blastp/results.xml")), desc='Reading blast results'):
        if record.alignments:
            query_gene_id = record.query.split('|')[-1]
            for align in record.alignments[1:]:
                for hsp in align.hsps:
                    #if hsp.expect < E_VALUE_THRESH:
                    homologue_gene_id = align.title.split('|')[-1]
                    if query_gene_id not in blast_results:
                        blast_results[query_gene_id] = list()
                    homologue_gene_ids = blast_results[query_gene_id]
                    homologue_gene_ids.append(homologue_gene_id)
                    blast_results[query_gene_id] = homologue_gene_ids
                    break

    blast_results = {key: value for key, value in blast_results.items() if value}
    num_sequence = sum(len(list) for list in gene_data.values())
    print('#gene_family:', len(gene_data), '#mum_sequence:', num_sequence, '#num_alignments:', len(blast_results))
    max_len = len(max([gene_sequence for gene_family, value in gene_data.items() for gene_id, gene_sequence in value], key=len))

    for gene_family, value in tqdm(gene_data.items(), desc='Encoding sequences ... '):
        encoded_sequences = [util.encode_sequence(id, seq, max_len) for id, seq in value]
        gene_data[gene_family] = encoded_sequences

    encoded_pairs = dict()
    for gene_family, encoded_sequences in tqdm(gene_data.items(), desc='Generating pairs ... '):
        encoded_sequences = [sequences for gene_id, sequences in encoded_sequences]
        encoded_pairs[gene_family] = util.generate_pairs(encoded_sequences)
    encoded_pairs = np.asarray([item for outer_list in encoded_pairs.values() for item in outer_list])

    train_pairs, validation_pairs = train_test_split(encoded_pairs, test_size=0.3, random_state=42)
    validation_pairs, test_pairs = train_test_split(validation_pairs, test_size=0.5, random_state=42)

    embed_dim = len(nucleotides)
    input_shape = encoded_pairs[0][0].shape
    input_a = Input(shape=input_shape, name='input_1')
    input_b = Input(shape=input_shape, name='input_2')
    embedding_layer = TokenAndPositionEmbedding(max_len=max_len, vocab_size=len(nucleotides), embed_dim=embed_dim)
    x1 = embedding_layer(input_a)
    x2 = embedding_layer(input_b)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    encoded_x1 = transformer_block(x1)
    encoded_x2 = transformer_block(x2)
    merged = concatenate([encoded_x1, encoded_x2])
    dropout = Dropout(0.1, name='dropout')(merged)
    hidden= Dense(hidden_layer_size, activation="relu", name='dense')(dropout)
    dropout2 = Dropout(0.1, name='dropout2')(hidden)

    average_pooled = GlobalAveragePooling2D(name='global_avg_pooling')(dropout2)
    output = Dense(1, activation="sigmoid", name='output')(average_pooled)
    supervised_model = Model(inputs=[input_a, input_b], outputs=output)

    output = Dense(4, activation="sigmoid", name='output')(dropout2)
    unsupervised_model = Model(inputs=[input_a, input_b], outputs=output)
    reconstruction_loss_a = tf.reduce_mean(tf.square(encoded_x1 - output))
    reconstruction_loss_b = tf.reduce_mean(tf.square(encoded_x2 - output))
    reconstruction_loss = (reconstruction_loss_a + reconstruction_loss_b) / 2

    plot_model(unsupervised_model, to_file='unsupervised_model.png', show_shapes=True, show_layer_names=True)
    plot_model(supervised_model, to_file='supervised_model.png', show_shapes=True, show_layer_names=True)

    supervised_model.add_loss(tf.keras.losses.binary_crossentropy)
    unsupervised_model.add_loss(reconstruction_loss)

    supervised_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    unsupervised_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    del encoded_pairs

    train_labels = list(np.ones(len(train_pairs), dtype=int))
    train_pairs, train_labels = util.add_noise(train_pairs, train_labels, gene_data)
    validation_labels = list(np.ones(len(validation_pairs), dtype=int))
    validation_pairs, validation_labels = util.add_noise(validation_pairs, validation_labels, gene_data)
    test_labels = list(np.ones(len(test_pairs), dtype=int))
    test_pairs, test_labels = util.add_noise(test_pairs, test_labels, gene_data)

    indices = np.arange(int(batch_size * int(max(1, len(train_pairs)/batch_size))))
    np.random.shuffle(indices)
    train_pairs = train_pairs[indices]
    train_labels = train_labels[indices]

    indices = np.arange(int(batch_size * int(max(1, len(validation_pairs)/batch_size))))
    np.random.shuffle(indices)
    validation_pairs = validation_pairs[indices]
    validation_labels = validation_labels[indices]

    print('#train:', len(train_pairs), '#validation:', len(validation_pairs), '#test:', len(test_pairs))

    data_reversed = dict()
    for gene_family, value in gene_data.items():
        for gene_id, seq in value:
            seq = ' '.join([' '.join(map(str, row)) for row in seq])
            seq_length = int(len(seq.split(' ')) / len(nucleotides))
            padding_x = ('0.0 ' * ((max_len - seq_length) * len(nucleotides))).rstrip(' ')
            if (len(padding_x)) > 0:
                padded_sequence = seq + ' ' + padding_x
            else:
                padded_sequence = seq
            data_reversed[padded_sequence] = (gene_family, gene_id)

    if model == 'unsupervised':

        for epoch in range(num_epochs):
            acc_group_loss = 0
            acc_group_losses = list()
            accumulated_gradients = [tf.zeros_like(var) for var in unsupervised_model.trainable_variables]
            train_dataset = tf.data.Dataset.from_tensor_slices((train_pairs[:, 0], train_pairs[:, 1]))
            train_dataset = train_dataset.batch(batch_size)
            train_iterator = iter(train_dataset)
            for batch_num in tqdm(range(len(train_dataset)), desc=f"Unsupervised Model Training: Epoch {epoch + 1}"):
                train_batch_data = next(train_iterator)
                with tf.GradientTape() as tape:
                    loss_value = unsupervised_model(train_batch_data)  # Assuming input_a and input_b are combined in batch_data
                gradients = tape.gradient(loss_value, unsupervised_model.trainable_variables)
                accumulated_gradients = [acc_grad + grad for acc_grad, grad in zip(accumulated_gradients, gradients)]

                acc_group_loss += loss_value

                if (batch_num + 1) % accumulation_steps == 0:
                    unsupervised_model.optimizer.apply_gradients(zip(accumulated_gradients, unsupervised_model.trainable_variables))
                    accumulated_gradients = [tf.zeros_like(var) for var in unsupervised_model.trainable_variables]
                    acc_group_loss = tf.reduce_mean(acc_group_loss, axis=(1, 2, 3))  # Calculate the average loss over dimensions 1, 2, 3
                    acc_group_loss = tf.reduce_mean(acc_group_loss)  # Calculate the final average loss
                    acc_group_losses.append(acc_group_loss)
                    average_train_loss = sum(acc_group_losses) / len(acc_group_losses)
                    acc_group_loss = 0

            exponent = int(math.log10(abs(average_train_loss)))
            coefficient = average_train_loss / (10 ** exponent)
            average_train_loss = "{:.2f} x 10^{}".format(coefficient, exponent)

            val_total_loss = 0
            val_dataset = tf.data.Dataset.from_tensor_slices((validation_pairs[:, 0], validation_pairs[:, 1]))
            val_dataset = val_dataset.batch(batch_size)
            val_iterator = iter(val_dataset)
            for batch_num in tqdm(range(len(val_dataset)), desc=f"Unsupervised Model Validation: Epoch {epoch + 1}"):
                val_batch_data = next(val_iterator)
                val_batch_loss = unsupervised_model(val_batch_data)  # Assuming input_a and input_b are combined in val_batch_data
                val_loss_value = tf.reduce_mean(val_batch_loss, axis=(1, 2, 3))  # Calculate the average loss over dimensions 1, 2, 3
                val_total_loss += tf.reduce_mean(val_loss_value)  # Calculate the final average validation loss
                average_val_loss = val_total_loss / (batch_num+1)

            exponent = int(math.log10(abs(average_val_loss)))
            coefficient = average_val_loss / (10 ** exponent)
            average_val_loss = "{:.2f} x 10^{}".format(coefficient, exponent)

            embeddings = unsupervised_model.predict([test_pairs[:, 0], test_pairs[:, 1]], batch_size=batch_size)

            def compute_similarity(embeddings):
                distances = embeddings[:, 0]
                for i in range(1, embeddings.shape[1]):
                    distances = distances - embeddings[:, i]
                normalized_distance1 = np.linalg.norm(distances, axis=1)
                normalized_distance2 = np.linalg.norm(normalized_distance1, axis=1)
                # Convert distances to similarity scores (higher is more similar)
                similarity_scores = 1 / (1 + normalized_distance2)
                return similarity_scores.tolist()

            similarity_scores = compute_similarity(embeddings)

            gene_similarity_dict = {}
            for i, test_pair in enumerate(test_pairs):
                sequence1 = ' '.join([' '.join(map(str, row)) for row in test_pair[0]])
                sequence2 = ' '.join([' '.join(map(str, row)) for row in test_pair[1]])
                score = similarity_scores[i]

                if sequence1 not in gene_similarity_dict or score > gene_similarity_dict[sequence1][1]:
                    gene_similarity_dict[sequence1] = (sequence2, score, i)

            correct_pred_mrna = 0
            correct_pred_prot = 0
            incorrect_pred_mrna = 0
            incorrect_pred_prot = 0
            for gene, (most_similar_gene, similarity_score, i) in gene_similarity_dict.items():
                gene_family1 = data_reversed[gene][0]
                gene_id1 = data_reversed[gene][1]
                gene_family2 = data_reversed[most_similar_gene][0]
                gene_id2 = data_reversed[most_similar_gene][1]
                if gene_family1 == gene_family2:
                    correct_pred_mrna += 1
                else:
                    incorrect_pred_mrna += 1
                try:
                    gene_id1_homologues = blast_results[gene_id1]
                    if gene_id2 in gene_id1_homologues:
                        correct_pred_prot += 1
                    else:
                        incorrect_pred_prot += 1
                except:
                    continue

            pred_protein = correct_pred_prot + incorrect_pred_prot
            correct_pred_prot = "{:.2f}".format((correct_pred_prot/pred_protein)*100)+'%'
            incorrect_pred_prot = "{:.2f}".format((incorrect_pred_prot/pred_protein)*100)+'%'
            correct_pred_mrna = "{:.2f}".format((correct_pred_mrna/len(gene_similarity_dict))*100)+'%'
            incorrect_pred_mrna = "{:.2f}".format((incorrect_pred_mrna/len(gene_similarity_dict))*100)+'%'

            embeddings = embeddings.reshape((embeddings.shape[0], embeddings.shape[1] * embeddings.shape[2]  * embeddings.shape[3]))

            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score, davies_bouldin_score

            kmeans = KMeans(n_clusters=2, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            silhouette = "{:.4f}".format(silhouette_score(embeddings, labels))
            davies_bouldin = "{:.4f}".format(davies_bouldin_score(embeddings, labels))

            from sklearn.neighbors import KernelDensity
            kde = KernelDensity()
            kde.fit(embeddings)
            # Generate new samples and calculate their log-density
            new_samples = kde.sample(len(embeddings))
            log_density = kde.score_samples(new_samples)
            log_density = sum(log_density) / len(log_density)
            exponent = int(math.log10(abs(log_density)))
            coefficient = log_density / (10 ** exponent)
            log_density = "{:.2f} x 10^{}".format(coefficient, exponent)

            from sklearn.cluster import KMeans
            cluster_variances = []
            for cluster in range(2):
                cluster_samples = embeddings[labels == cluster]
                variance = np.var(cluster_samples)
                cluster_variances.append(variance)
            mean_cluster_variance = np.mean(cluster_variances)
            exponent = int(math.log10(abs(mean_cluster_variance)))
            coefficient = mean_cluster_variance / (10 ** exponent)
            mean_cluster_variance = "{:.2f} x 10^{}".format(coefficient, exponent)

            print(f"Unsupervised Model Epoch {epoch+1}: Training Loss {average_train_loss}, Validation Loss {average_val_loss}, size of gene_similarity_dict {len(gene_similarity_dict)} #correct_pred(mRNA) {correct_pred_mrna} #incorrect_pred(mRNA) {incorrect_pred_mrna}, #correct_pred(prot) {correct_pred_prot} #incorrect_pred(prot) {incorrect_pred_prot}, Silhouette Score: {silhouette} Average Log-Density: {log_density} Mean Intra-cluster Variance: {mean_cluster_variance}, Davies Bouldin: {davies_bouldin}")

    elif model == 'supervised':

        for epoch in range(num_epochs):
            acc_group_loss = 0
            acc_group_losses = list()
            accumulated_gradients = [tf.zeros_like(var) for var in supervised_model.trainable_variables]
            train_dataset = tf.data.Dataset.from_tensor_slices(((train_pairs[:, 0], train_pairs[:, 1]), train_labels))
            train_dataset = train_dataset.batch(batch_size)
            train_iterator = iter(train_dataset)
            for batch_num in tqdm(range(len(train_dataset)), desc=f"Supervised Model Training: Epoch {epoch + 1}"):
                train_batch_data, train_batch_labels = next(train_iterator)
                with tf.GradientTape() as tape:
                    predictions = supervised_model(train_batch_data, training=True)
                    train_batch_labels = tf.reshape(train_batch_labels, (len(train_batch_labels), 1))
                    loss_value = tf.keras.losses.binary_crossentropy(train_batch_labels, predictions)
                gradients = tape.gradient(loss_value, supervised_model.trainable_variables)
                accumulated_gradients = [acc_grad + grad for acc_grad, grad in zip(accumulated_gradients, gradients)]

                acc_group_loss += tf.reduce_mean(loss_value)

                if (batch_num + 1) % accumulation_steps == 0:
                    supervised_model.optimizer.apply_gradients(zip(accumulated_gradients, supervised_model.trainable_variables))
                    accumulated_gradients = [tf.zeros_like(var) for var in supervised_model.trainable_variables]
                    acc_group_loss = acc_group_loss / accumulation_steps
                    acc_group_losses.append(acc_group_loss)
                    average_train_loss = sum(acc_group_losses) / len(acc_group_losses)
                    acc_group_loss = 0

            exponent = int(math.log10(abs(average_train_loss)))
            coefficient = average_train_loss / (10 ** exponent)
            average_train_loss = "{:.2f} x 10^{}".format(coefficient, exponent)

            val_total_loss = 0
            val_dataset = tf.data.Dataset.from_tensor_slices(((validation_pairs[:, 0], validation_pairs[:, 1]), validation_labels))
            val_dataset = val_dataset.batch(batch_size)
            val_iterator = iter(val_dataset)
            for batch_num in tqdm(range(len(val_dataset)), desc=f"Supervised Model Validation: Epoch {epoch + 1}"):
                val_batch_data, val_batch_labels = next(val_iterator)
                val_predictions = supervised_model(val_batch_data, training=False)
                val_batch_labels = tf.reshape(val_batch_labels, (len(val_batch_labels), 1))
                #val_predictions = tf.reshape(val_predictions, len(val_predictions))
                val_batch_loss = tf.keras.losses.binary_crossentropy(val_batch_labels, val_predictions)
                val_batch_loss = tf.reduce_mean(val_batch_loss)
                val_total_loss += val_batch_loss
                average_val_loss = val_total_loss / (batch_num+1)

            exponent = int(math.log10(abs(average_val_loss)))
            coefficient = average_val_loss / (10 ** exponent)
            average_val_loss = "{:.2f} x 10^{}".format(coefficient, exponent)

            y_pred = supervised_model.predict([test_pairs[:, 0], test_pairs[:, 1]], batch_size=batch_size).reshape(len(test_pairs))
            y_true = test_labels
            threshold = 0.5
            y_pred_binary = np.where(y_pred >= threshold, 1, 0)
            tp = np.sum(np.logical_and(y_pred_binary == 1, y_true == 1))
            tn = np.sum(np.logical_and(y_pred_binary == 0, y_true == 0))
            fp = np.sum(np.logical_and(y_pred_binary == 1, y_true == 0))
            fn = np.sum(np.logical_and(y_pred_binary == 0, y_true == 1))

            correct_pred_mrna = 0
            correct_pred_prot = 0
            incorrect_pred_mrna = 0
            incorrect_pred_prot = 0

            for i, test_pair in enumerate(test_pairs):
                sequence1 = ' '.join([' '.join(map(str, row)) for row in test_pair[0]])
                sequence2 = ' '.join([' '.join(map(str, row)) for row in test_pair[1]])
                gene_family1, gene_id1 = data_reversed[sequence1]
                gene_family2, gene_id2 = data_reversed[sequence2]
                if y_true[i] == y_pred_binary[i]:
                    correct_pred_mrna += 1
                    if y_true[i] == 1 and y_pred_binary[i] == 1:
                        try:
                            gene_id1_homologues = blast_results[gene_id1]
                            if gene_id2 in gene_id1_homologues:
                                correct_pred_prot += 1
                            else:
                                incorrect_pred_prot += 1
                        except:
                            continue
                else:
                    incorrect_pred_mrna += 1

            pred_protein = correct_pred_prot + incorrect_pred_prot
            correct_pred_prot = "{:.2f}".format((correct_pred_prot/pred_protein)*100)+'%'
            incorrect_pred_prot = "{:.2f}".format((incorrect_pred_prot/pred_protein)*100)+'%'
            correct_pred_mrna = "{:.2f}".format((correct_pred_mrna/len(test_pairs))*100)+'%'
            incorrect_pred_mrna = "{:.2f}".format((incorrect_pred_mrna/len(test_pairs))*100)+'%'

            accuracy = "{:.2f}".format((tp + tn) / (tp + tn + fp + fn))
            sensitivity = "{:.2f}".format(tp / (tp + fn))
            specificity = "{:.2f}".format(tn / (tn + fp))

            print(f"Supervised Model Epoch {epoch+1}: Training Loss {average_train_loss}, Validation Loss {average_val_loss}, #tp {tp} #tn {tn} #fp {fp} #fn {fn},  #correct_pred(mRNA) {correct_pred_mrna} #incorrect_pred(mRNA) {incorrect_pred_mrna}, #correct_pred(prot) {correct_pred_prot} #incorrect_pred(prot) {incorrect_pred_prot}, Accuracy: {accuracy} Sensitivity: {sensitivity} Specificity: {specificity}")

    else:
        raise Exception('model type should be supervised or unsupervised...')



