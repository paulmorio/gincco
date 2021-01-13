"""Utility functions

A number of functions for loading and transforming data used in the project
"""

# Standard
import os

# 3rd Party
from tqdm import tqdm
import networkx as nx
from networkx.algorithms.components import connected_components

def get_metabric_genes(fh):
    """ Returns a list of the genes in the metabric csv file header
    NB: It removes the "GE_" suffix present in the naming scheme
    used for gene expression

    Parameters
    ----------
    fh : str
        path to the metabric csv file

    Returns
    -------
    metabric_genes : list
        list of metabric gene names without the GE_prefix
    """

    with open(fh, "r") as metabric_fh:
        header = (metabric_fh.readline()).strip()
        header = header.split(",")

    # Find all the genes in the header
    metabric_genes = []
    for var in header:
        if "GE_" in var:
            metabric_genes.append(var[3:])

    return metabric_genes

def get_stringdb_ppi_edge_list(string_meta_fh, string_ppi_fh):
    """Returns the list of tuples specifying the edge list that describes
    the PPI network. It replaces the identifiers used in the STRING database
    with the common gene name used in metabric (and in general publications)
    
    Parameters
    ----------
    string_meta_fh : str
        path to "info" file for string ppi data
    string_ppi_fh : str
        path to "links" file for string ppi data

    Returns
    -------
    ppi_edge_list: list of tuples 
        edge list in the format [(prot1, prot2), ...]
    """

    # load STRING protein meta-information
    print ("## Loading STRING protein meta-data")
    with open(string_meta_fh, "r") as pinfo_fh:
        stringdb_id2name = {}
        protein_meta_info = {}
        pinfo = pinfo_fh.readlines()
        for line in pinfo:
            line_split = (line.strip()).split("\t")
            string_db_id, real_prot_name, prot_size, prot_description = line_split
            stringdb_id2name[string_db_id] = real_prot_name
            protein_meta_info[real_prot_name] = [prot_size, prot_description]

    # load STRING PPI network and change with real_prot_names
    print("## Matching STRING identifiers with real protein names")
    with open(string_ppi_fh, "r") as ppi_fh:
        ppi_edge_list = []
        ppi_weighted_edge_list = []
        stringdb_edges = ppi_fh.readlines()
        for line in tqdm(stringdb_edges[1:]):
            edge_stringdb = (line.strip()).split(" ")
            prot1, prot2, combo_score = edge_stringdb
            ppi_edge_list.append((stringdb_id2name[prot1], stringdb_id2name[prot2]))

    return ppi_edge_list


def get_weighted_stringdb_ppi_edge_list(string_meta_fh, string_ppi_fh):
    """
    Returns the list of tuples specifying the edge list that describes
    the PPI network. It replaces the identifiers used in the STRING database
    with the common gene name used in metabric (and in general publications)

    This tuple contains 3 items, specifying the two connected proteins and the associated
    combined score as computed by the STRING database. (prot1, prot2, weight)

    Returns
    ppi_weighted_edge_list: list of tuples, format [(prot1, prot2, weight), ...]
    """

    # load STRING protein meta-information
    print ("## Loading STRING protein meta-data")
    with open(string_meta_fh, "r") as pinfo_fh:
        stringdb_id2name = {}
        protein_meta_info = {}
        pinfo = pinfo_fh.readlines()
        for line in pinfo:
            line_split = (line.strip()).split("\t")
            string_db_id, real_prot_name, prot_size, prot_description = line_split
            stringdb_id2name[string_db_id] = real_prot_name
            protein_meta_info[real_prot_name] = [prot_size, prot_description]

    # load STRING PPI network and change with real_prot_names
    print("## Matching STRING identifiers with real protein names")
    with open(string_ppi_fh, "r") as ppi_fh:
        ppi_weighted_edge_list = []
        stringdb_edges = ppi_fh.readlines()
        for line in tqdm(stringdb_edges[1:]):
            edge_stringdb = (line.strip()).split(" ")
            prot1, prot2, combo_score = edge_stringdb
            ppi_weighted_edge_list.append((stringdb_id2name[prot1], stringdb_id2name[prot2], combo_score))
    print("## Done ...")

    return ppi_weighted_edge_list


def get_metabric_ppi_network(metabric_filehandle, string_meta_fh, string_ppi_fh, weighted=False):
    """Returns a NetworkX graph of induced subgraph of the PPI network
    intersecting the PPI network and the supplied metabric genes

    Parameters
    ----------
    metabric_filehandle : str
    string_meta_fh : str
    string_ppi_fh : str
    weighted : bool

    Returns
    -------
    G : nx.Graph
        The induced subgraph containing the common genes in the metabric
        gene expression data and that of the input network (stringdb). Not
        connected (so GNNs wont work on this)
    """

    metabric_genes = get_metabric_genes(metabric_filehandle)

    # find weighted or unweighted subnetwork
    if weighted:
        ppi_edge_list = get_weighted_stringdb_ppi_edge_list(string_meta_fh, string_ppi_fh)
        metabric_ppi_edge_list = []
        ppi_edge_list_len = len(ppi_edge_list)
        print("## Finding metabric genes subgraph in STRING network via weighted edge list")
        for edge in tqdm(ppi_edge_list):
            if edge[0] in metabric_genes and edge[1] in metabric_genes:
                metabric_ppi_edge_list.append(edge)

        # Create NetworkX graph of the METABRIC PPI network
        print("## Generating NetworkX Graph of METABRIC PPI network")
        G = nx.Graph()
        G.add_weighted_edges_from(metabric_ppi_edge_list)

    else:
        ppi_edge_list = get_stringdb_ppi_edge_list(string_meta_fh, string_ppi_fh)
        metabric_ppi_edge_list = []
        ppi_edge_list_len = len(ppi_edge_list)
        print("## Finding metabric genes subgraph in STRING network via edge list")
        for edge in tqdm(ppi_edge_list):
            if edge[0] in metabric_genes and edge[1] in metabric_genes:
                metabric_ppi_edge_list.append(edge)

        # Create NetworkX graph of the METABRIC PPI network
        print("## Generating NetworkX Graph of METABRIC PPI network")
        G = nx.Graph()
        G.add_edges_from(metabric_ppi_edge_list)

    return G


def get_generic_string_overlap_network(gene_symbols, string_meta_fh, string_ppi_fh, weighted=False):
    

    # find weighted or unweighted subnetwork
    if weighted:
        ppi_edge_list = get_weighted_stringdb_ppi_edge_list(string_meta_fh, string_ppi_fh)
        metabric_ppi_edge_list = []
        ppi_edge_list_len = len(ppi_edge_list)
        print("## Finding intersecting genes subgraph in STRING network via weighted edge list")
        for edge in tqdm(ppi_edge_list):
            if edge[0] in gene_symbols and edge[1] in gene_symbols:
                metabric_ppi_edge_list.append(edge)

        # Create NetworkX graph of the METABRIC PPI network
        print("## Generating NetworkX SubGraph of String PPI network")
        G = nx.Graph()
        G.add_weighted_edges_from(metabric_ppi_edge_list)

    else:
        ppi_edge_list = get_stringdb_ppi_edge_list(string_meta_fh, string_ppi_fh)
        metabric_ppi_edge_list = []
        ppi_edge_list_len = len(ppi_edge_list)
        print("## Finding intersecting genes subgraph in STRING network via edge list")
        for edge in tqdm(ppi_edge_list):
            if edge[0] in gene_symbols and edge[1] in gene_symbols:
                metabric_ppi_edge_list.append(edge)

        # Create NetworkX graph of the METABRIC PPI network
        print("## Generating NetworkX SubGraph of String PPI network")
        G = nx.Graph()
        G.add_edges_from(metabric_ppi_edge_list)

    return G


def get_connected_metabric_ppi_network(metabric_filehandle, string_meta_fh, string_ppi_fh, weighted=False):
    """Returns a NetworkX graph of the largest connected PPI network at the 
    intersection of the gene sets available in the metabric dataset and the 
    STRING (human) ppi network.
    
    Parameters
    ----------
    metabric_file_handle : str
    string_meta_fh : str
    string_ppi_fh : str
    weighted : bool
    
    Returns
    -------
    largest_cc : nx.Graph
        Largest connected component. The largest connected PPI network at the 
        intersection of the gene sets available in the metabric dataset and the 
        STRING (human) ppi network.
    """

    metabric_genes = get_metabric_genes(metabric_filehandle)

    # Weighted PPI network (by STRING combined score)
    if weighted:
        ppi_edge_list = get_weighted_stringdb_ppi_edge_list(string_meta_fh, string_ppi_fh)
        metabric_ppi_edge_list = []
        ppi_edge_list_len = len(ppi_edge_list)
        for edge in tqdm(ppi_edge_list):
            if edge[0] in metabric_genes and edge[1] in metabric_genes:
                metabric_ppi_edge_list.append(edge)

        # Create NetworkX graph of the METABRIC PPI network
        print("## Generating NetworkX Graph of METABRIC PPI network")
        G = nx.Graph()
        G.add_weighted_edges_from(metabric_ppi_edge_list)
        print("Finding maximal connected component of METABRIC PPI network")
        largest_cc = max(connected_components(G), key=len)
        largest_cc = G.subgraph(largest_cc).copy()
    
    # Unweighted PPI network
    else:
        ppi_edge_list = get_stringdb_ppi_edge_list(string_meta_fh, string_ppi_fh)
        metabric_ppi_edge_list = []
        ppi_edge_list_len = len(ppi_edge_list)
        print("## Finding metabric genes subgraph in STRING network via edge list")
        for edge in tqdm(ppi_edge_list):
            if edge[0] in metabric_genes and edge[1] in metabric_genes:
                metabric_ppi_edge_list.append(edge)

        # Create NetworkX graph of the METABRIC PPI network
        print("## Generating NetworkX Graph of METABRIC PPI network")
        G = nx.Graph()
        G.add_edges_from(metabric_ppi_edge_list)
        print("Finding maximal connected component of METABRIC PPI network")
        largest_cc = max(connected_components(G), key=len)
        largest_cc = G.subgraph(largest_cc).copy()

    return largest_cc
