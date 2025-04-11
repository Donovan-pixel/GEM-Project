import xml.etree.ElementTree as Et
from collections import defaultdict
from os import listdir
import json
from xml.dom import minidom
import copy


class Triple:
    """
    Represents a single RDF triple with subject (s), predicate (p), and object (o).
    Triples are the basic building blocks of semantic web data.
    """

    def __init__(self, s, p, o):
        """
        Initialize a triple with subject, predicate, and object.
        
        Args:
            s (str): Subject - typically an entity
            p (str): Predicate - the relationship between subject and object
            o (str): Object - typically another entity or a value
        """
        self.s = s
        self.o = o
        self.p = p

    def flat_triple(self):
        """
        Returns a string representation of the triple with components separated by ' | '.
        
        Returns:
            str: Flattened string representation of the triple
        """
        return self.s + ' | ' + self.p + ' | ' + self.o


class Tripleset:
    """
    A collection of Triple objects. Used to represent related facts about entities.
    May represent either original or modified triples.
    """

    def __init__(self):
        """
        Initialize an empty Tripleset.
        The clusterid is used for grouping related triplesets.
        """
        self.triples = []  # List to store Triple objects
        self.clusterid = 0  # ID for clustering related triplesets

    def fill_tripleset(self, t):
        """
        Parse XML triple elements and fill the tripleset with Triple objects.
        
        Args:
            t: XML elements containing triple data in "s | p | o" format
        """
        for xml_triple in t:
            s, p, o = xml_triple.text.split(' | ')
            triple = Triple(s, p, o)
            self.triples.append(triple)


class Lexicalisation:
    """
    Represents a natural language expression of a Tripleset.
    One Tripleset can have multiple lexicalisations in different languages or styles.
    """

    def __init__(self, lex, lid, comment='', lang=''):
        """
        Initialize a lexicalisation.
        
        Args:
            lex (str): The text of the lexicalisation
            lid (str): Lexicalisation identifier
            comment (str, optional): Additional notes about the lexicalisation
            lang (str, optional): Language code of the lexicalisation
        """
        self.lex = lex  # The text of the lexicalisation
        self.id = lid   # Unique identifier
        self.comment = comment  # Optional comments
        self.lang = lang  # Language of the lexicalisation

    def chars_length(self):
        """
        Calculate the length of the lexicalisation in characters.
        
        Returns:
            int: Number of characters in the lexicalisation
        """
        return len(self.lex)


class Entry:
    """
    Represents a complete benchmark entry containing triples and their lexicalisations.
    An entry is the main unit in the benchmark dataset.
    """

    def __init__(self, category, size, eid, shape, shape_type):
        """
        Initialize a benchmark entry.
        
        Args:
            category (str): Domain category of the entry (e.g., 'Airport', 'Person')
            size (str): Number of triples in the entry
            eid (str): Entry identifier
            shape (str): Shape of the RDF graph
            shape_type (str): Type of shape
        """
        self.category = category  # Domain category of the content
        self.size = size  # Number of triples
        self.id = eid  # Entry identifier
        self.shape = shape  # Graph shape 
        self.shape_type = shape_type  # Type of graph shape
        self.originaltripleset = []  # List of original triple sets
        self.modifiedtripleset = Tripleset()  # Modified version of triples
        self.lexs = []  # List of lexicalisations
        self.dbpedialinks = []  # Links to DBpedia resources
        self.links = []  # Other relevant links

    def fill_originaltriple(self, xml_t):
        """
        Fill the original tripleset from XML data.
        Note that an entry can have multiple original triplesets.
        
        Args:
            xml_t: XML elements containing original triple data
        """
        otripleset = Tripleset()
        self.originaltripleset.append(otripleset)   # multiple originaltriplesets for one entry
        otripleset.fill_tripleset(xml_t)

    def fill_modifiedtriple(self, xml_t):
        """
        Fill the modified tripleset from XML data.
        Modified triples are typically normalized versions of original triples.
        
        Args:
            xml_t: XML elements containing modified triple data
        """
        self.modifiedtripleset.fill_tripleset(xml_t)

    def create_lex(self, xml_lex):
        """
        Create a lexicalisation from XML data and add it to the entry.
        
        Args:
            xml_lex: XML element containing lexicalisation data
        """
        try:
            comment = xml_lex.attrib['comment']
        except KeyError:
            comment = ''
        try:
            lang = xml_lex.attrib['lang']
        except KeyError:
            lang = ''
        lid = xml_lex.attrib['lid']
        lex = Lexicalisation(xml_lex.text, lid, comment, lang)
        self.lexs.append(lex)

    def create_dbpedialinks(self, xml_dbpedialinks):
        """
        Create DBpedia links from XML data and add them to the entry.
        These links connect entities to DBpedia resources.
        
        Args:
            xml_dbpedialinks: XML elements containing DBpedia link data
        """
        for xml_dblink in xml_dbpedialinks:
            s, p, o = xml_dblink.text.split(' | ')
            dbp_link = Triple(s, p, o)
            self.dbpedialinks.append(dbp_link)

    def create_links(self, xml_links):
        """
        Create general links from XML data and add them to the entry.
        These can be links to other resources or datasets.
        
        Args:
            xml_links: XML elements containing link data
        """
        for xml_link in xml_links:
            s, p, o = xml_link.text.split(' | ')
            link = Triple(s, p, o)
            self.links.append(link)

    def count_lexs(self):
        """
        Count the number of lexicalisations for this entry.
        
        Returns:
            int: Number of lexicalisations
        """
        return len(self.lexs)

    def flat_tripleset(self):
        """
        Render modified triples to a flat representation with <br> separators.
        Useful for display or comparison purposes.
        
        Returns:
            str: Flat representation of the tripleset
        """
        flat_mr = []
        for triple in self.modifiedtripleset.triples:
            flat_triple = triple.s + ' | ' + triple.p + ' | ' + triple.o
            flat_mr.append(flat_triple)
        if self.size == '1':
            return flat_mr[0]
        else:
            return '<br>'.join(flat_mr)

    def relations(self):
        """
        Extract the set of unique predicates (relations) used in the tripleset.
        
        Returns:
            set: Set of unique predicates
        """
        rel_set = set()
        for triple in self.modifiedtripleset.triples:
            rel_set.add(triple.p)
        return rel_set

    def list_triples(self):
        """
        Return a list of flattened triple strings for this entry.
        
        Returns:
            list: List of triple strings in format "s | p | o"
        """
        triples = []
        for triple in self.modifiedtripleset.triples:
            flat_triple = triple.s + ' | ' + triple.p + ' | ' + triple.o
            triples.append(flat_triple)
        return triples


class Benchmark:
    """
    Top-level class representing the entire benchmark dataset.
    Contains methods for dataset manipulation, filtering, and analysis.
    """

    def __init__(self):
        """
        Initialize an empty benchmark.
        """
        self.entries = []  # List to store Entry objects

    def fill_benchmark(self, fileslist):
        """
        Parse XML files and populate the benchmark with Entry instances.
        This is the main method for loading data into the benchmark.
        
        Args:
            fileslist: List of tuples (path_to_file, filename.xml)
        """
        for file in fileslist:
            myfile = file[0] + '/' + file[1]
            tree = Et.parse(myfile)
            root = tree.getroot()
            for xml_entry in root.iter('entry'):
                # Extract entry attributes
                entry_id = xml_entry.attrib['eid']
                category = xml_entry.attrib['category']
                size = xml_entry.attrib['size']
                shape = xml_entry.attrib['shape']
                shape_type = xml_entry.attrib['shape_type']

                # Create new entry
                entry = Entry(category, size, entry_id, shape, shape_type)
                
                # Fill entry with data from XML
                for child in xml_entry:
                    if child.tag == 'originaltripleset':
                        entry.fill_originaltriple(child)
                    elif child.tag == 'modifiedtripleset':
                        entry.fill_modifiedtriple(child)
                    elif child.tag == 'lex':
                        entry.create_lex(child)
                    elif child.tag == 'dbpedialinks':
                        entry.create_dbpedialinks(child)
                    elif child.tag == 'links':
                        entry.create_links(child)
                
                # Add completed entry to benchmark
                self.entries.append(entry)

    def total_lexcount(self):
        """
        Calculate the total number of lexicalisations across all entries.
        
        Returns:
            int: Total number of lexicalisations
        """
        count = [entry.count_lexs() for entry in self.entries]
        return sum(count)

    def unique_p_otriples(self):
        """
        Extract the set of unique predicates from original triples.
        
        Returns:
            set: Set of unique predicates from original triples
        """
        properties = [triple.p for entry in self.entries for triple in entry.originaltripleset[0].triples]
        return set(properties)

    def unique_p_mtriples(self):
        """
        Extract the set of unique predicates from modified triples.
        
        Returns:
            set: Set of unique predicates from modified triples
        """
        properties = [triple.p for entry in self.entries for triple in entry.modifiedtripleset.triples]
        return set(properties)

    def entry_count(self, size=None, cat=None):
        """
        Count entries matching specific criteria.
        
        Args:
            size (str, optional): Size filter
            cat (str, optional): Category filter
            
        Returns:
            int: Number of matching entries
        """
        if not size and cat:
            entries = [entry for entry in self.entries if entry.category == cat]
        elif not cat and size:
            entries = [entry for entry in self.entries if entry.size == size]
        elif not size and not cat:
            return len(self.entries)
        else:
            entries = [entry for entry in self.entries if entry.category == cat and entry.size == size]
        return len(entries)

    def lexcount_size_category(self, size, cat):
        """
        Calculate the number of lexicalisations for entries with specific size and category.
        
        Args:
            size (str): Size filter
            cat (str): Category filter
            
        Returns:
            int: Total number of lexicalisations for matching entries
        """
        counts = [entry.count_lexs() for entry in self.entries if entry.category == cat and entry.size == size]
        return sum(counts)

    def property_map(self):
        """
        Create an approximate mapping between modified properties and original properties.
        This helps track how predicates are normalized between original and modified triples.
        
        Returns:
            defaultdict: Mapping from modified properties to sets of original properties
        """
        mprop_oprop = defaultdict(set)
        for entry in self.entries:
            for tripleset in entry.originaltripleset:
                for i, triple in enumerate(tripleset.triples):
                    m_property = entry.modifiedtripleset.triples[i].p
                    m_subj = entry.modifiedtripleset.triples[i].s
                    m_obj = entry.modifiedtripleset.triples[i].o
                    if m_subj == triple.s and m_obj == triple.o:  # Match based on subject and object
                        mprop_oprop[m_property].add(triple.p)
                    if not mprop_oprop[m_property]:  # Fallback mapping
                        mprop_oprop[m_property].add(triple.p)
        return mprop_oprop

    def filter(self, size=[], cat=[]):
        """
        Filter entries based on size and category criteria.
        Returns a new benchmark object containing only the filtered entries.
        
        Args:
            size (list): List of triple sizes to include
            cat (list): List of categories to include
            
        Returns:
            Benchmark: New benchmark object with filtered entries, or None if empty
        """
        bench_filtered = self.copy()
        for entry in self.entries:
            deleted = False
            if cat:
                if entry.category not in cat:
                    bench_filtered.del_entry(entry)
                    deleted = True
            if size and not deleted:
                if entry.size not in size:
                    bench_filtered.del_entry(entry)
        if bench_filtered.entries:
            return bench_filtered
        else:
            return None

    def copy(self):
        """
        Create a deep copy of the benchmark.
        Useful for creating filtered views without modifying the original.
        
        Returns:
            Benchmark: Deep copy of this benchmark
        """
        b_copy = Benchmark()
        b_copy.entries = copy.deepcopy(self.entries)
        return b_copy

    def filter_by_entry_ids(self, entry_ids):
        """
        Filter entries based on their IDs.
        Returns a new benchmark object containing only entries with specified IDs.
        
        Args:
            entry_ids (list): List of entry IDs to include
            
        Returns:
            Benchmark: New benchmark object with filtered entries
        """
        bench_filtered = self.copy()
        for entry in self.entries:
            if entry.id not in entry_ids:
                bench_filtered.del_entry(entry)
        return bench_filtered

    def triplesets(self):
        """
        Extract all modified triplesets from the benchmark.
        
        Returns:
            list: List of all Tripleset objects
        """
        all_triplesets = [entry.modifiedtripleset for entry in self.entries]
        return all_triplesets

    def del_entry(self, entry):
        """
        Delete an entry from the benchmark.
        Uses entry ID to match entries, which is useful when deleting from a copy.
        
        Args:
            entry (Entry): Entry to delete (matching by ID)
        """
        for init_entry in self.entries:
            if init_entry.id == entry.id:
                self.entries.remove(init_entry)

    def get_lex_by_id(self, entry_category, entry_size, entry_id, lex_id):
        """
        Retrieve a specific lexicalisation by entry and lexicalisation IDs.
        
        Args:
            entry_category (str): Category of the entry
            entry_size (str): Size of the entry
            entry_id (str): ID of the entry
            lex_id (str): ID of the lexicalisation
            
        Returns:
            str: The text of the lexicalisation, or None if not found
        """
        for entry in self.entries:
            if entry.id == entry_id and entry.size == entry_size and entry.category == entry_category:
                for lex in entry.lexs:
                    if lex.id == lex_id:
                        return lex.lex

    def subjects_objects(self):
        """
        Extract all unique subjects and objects from original triples.
        
        Returns:
            tuple: (set of subjects, set of objects)
        """
        subjects = set()
        objects = set()
        for entry in self.entries:
            for triple in entry.originaltripleset[0].triples:
                subjects.add(triple.s)
                objects.add(triple.o)
        return subjects, objects

    def verbalisations(self):
        """
        Extract all lexicalisations (verbalisations) from the benchmark.
        
        Returns:
            list: List of all lexicalisation texts
        """
        verbalisations = []
        for entry in self.entries:
            for lex in entry.lexs:
                verbalisations.append(lex.lex)
        return verbalisations

    def sort_by_size_and_name(self):
        """
        Sort entries by size, then by flattened triple representation.
        Modifies the benchmark in-place.
        
        Returns:
            Benchmark: Self, with entries sorted
        """
        sorted_entries = sorted(self.entries, key=lambda x: (x.size, x.flat_tripleset()))
        self.entries = sorted_entries
        return self

    def b2json(self, path, filename):
        """
        Convert benchmark to JSON format and save to file.
        Creates a new JSON structure with all benchmark data.
        
        Args:
            path (str): Directory path to save the file
            filename (str): Name of the output JSON file
        """
        data = {}
        data['entries'] = []
        entry_id = 0  # new entry ids
        for entry in self.entries:
            entry_id += 1
            # Create data structures for JSON
            orig_triplesets = {}
            orig_triplesets['originaltripleset'] = []
            modif_tripleset = []
            lexs = []
            links = []
            dbpedialinks = []
            
            # Fill original triplesets
            for otripleset in entry.originaltripleset:
                orig_tripleset = []
                for triple in otripleset.triples:
                    orig_tripleset.append({'subject': triple.s, 'property': triple.p, 'object': triple.o})
                orig_triplesets['originaltripleset'].append(orig_tripleset)
            
            # Fill modified tripleset
            for triple in entry.modifiedtripleset.triples:
                modif_tripleset.append({'subject': triple.s, 'property': triple.p, 'object': triple.o})
            
            # Fill lexicalisations
            for lex in entry.lexs:
                lexs.append({'comment': lex.comment, 'xml_id': lex.id, 'lex': lex.lex, 'lang': lex.lang})
            
            # Fill DBpedia links
            if entry.dbpedialinks:
                for link in entry.dbpedialinks:
                    dbpedialinks.append({'subject': link.s, 'property': link.p, 'object': link.o})
            
            # Fill other links
            if entry.links:
                for link in entry.links:
                    links.append({'subject': link.s, 'property': link.p, 'object': link.o})
            
            # Add complete entry to JSON data
            data['entries'].append({entry_id: {'category': entry.category, 'size': entry.size, 'xml_id': entry.id,
                                               'shape': entry.shape, 'shape_type': entry.shape_type,
                                               'originaltriplesets': orig_triplesets,
                                               'modifiedtripleset': modif_tripleset,
                                               'lexicalisations': lexs,
                                               'dbpedialinks': dbpedialinks,
                                               'links': links}
                                    })
        
        # Write JSON to file
        with open(path + '/' + filename, 'w+', encoding='utf8') as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4, sort_keys=True)

    def b2xml(self, path, filename, recalc_id=True):
        """
        Convert benchmark to XML format and save to file.
        Creates a new XML structure with all benchmark data.
        
        Args:
            path (str): Directory path to save the file
            filename (str): Name of the output XML file
            recalc_id (bool): Whether to recalculate entry IDs
        """
        # Create root XML element
        root = Et.Element('benchmark')
        entries_xml = Et.SubElement(root, 'entries')
        
        # Process each entry
        for index, entry in enumerate(self.entries):
            if recalc_id:
                entry.id = str(index + 1)  # Recalculate entry ID if requested
            
            # Create entry element with attributes
            entry_xml = Et.SubElement(entries_xml, 'entry',
                                      attrib={'category': entry.category, 'eid': entry.id, 'size': entry.size,
                                              'shape': entry.shape, 'shape_type': entry.shape_type})
            
            # Add originaltripleset elements
            for otripleset in entry.originaltripleset:
                otripleset_xml = Et.SubElement(entry_xml, 'originaltripleset')
                for triple in otripleset.triples:
                    otriple_xml = Et.SubElement(otripleset_xml, 'otriple')
                    otriple_xml.text = triple.s + ' | ' + triple.p + ' | ' + triple.o
            
            # Add modifiedtripleset element
            mtripleset_xml = Et.SubElement(entry_xml, 'modifiedtripleset')
            for mtriple in entry.modifiedtripleset.triples:
                mtriple_xml = Et.SubElement(mtripleset_xml, 'mtriple')
                mtriple_xml.text = mtriple.s + ' | ' + mtriple.p + ' | ' + mtriple.o
            
            # Add lexicalisation elements
            for lex in entry.lexs:
                lex_xml = Et.SubElement(entry_xml, 'lex', attrib={'comment': lex.comment, 'lid': lex.id,
                                                                  'lang': lex.lang})
                lex_xml.text = lex.lex
            
            # Add DBpedia links if any
            if entry.dbpedialinks:
                dbpedialinks_xml = Et.SubElement(entry_xml, 'dbpedialinks')
                for link in entry.dbpedialinks:
                    dbpedialink_xml = Et.SubElement(dbpedialinks_xml, 'dbpedialink', attrib={'direction': 'en2ru'})
                    dbpedialink_xml.text = link.s + ' | ' + link.p + ' | ' + link.o
            
            # Add other links if any
            if entry.links:
                links_xml = Et.SubElement(entry_xml, 'links')
                for link in entry.links:
                    link_xml = Et.SubElement(links_xml, 'link', attrib={'direction': 'en2ru'})
                    link_xml.text = link.s + ' | ' + link.p + ' | ' + link.o
        
        # Convert XML element tree to pretty-printed string
        ugly_xml_string = Et.tostring(root, encoding='utf-8', method='xml')
        xml = minidom.parseString(ugly_xml_string).toprettyxml(indent='  ')
        
        # Write to file
        with open(path + '/' + filename, 'w+', encoding='utf-8') as f:
            f.write(xml)

    @staticmethod
    def categories():
        """
        Return a list of standard domain categories used in the benchmark.
        
        Returns:
            list: List of category names
        """
        return ['Airport', 'Artist', 'Astronaut', 'Athlete', 'Building', 'CelestialBody', 'City',
                'ComicsCharacter', 'Company', 'Food', 'MeanOfTransportation', 'Monument',
                'Politician', 'SportsTeam', 'University', 'WrittenWork']


def select_files(topdir, category='', size=(1, 8)):
    """
    Helper function to select XML files from a directory structure.
    Searches for files in specific directories based on category and size.
    
    Args:
        topdir (str): Top directory containing triples subdirectories
        category (str, optional): Category filter for file selection
        size (tuple, optional): Range of triple sizes to include (min, max)
        
    Returns:
        list: List of tuples (directory, filename) for the selected files
    """
    # Create a list of directories to search (e.g., '1triples', '2triples', etc.)
    finaldirs = [topdir+'/'+str(item)+'triples' for item in range(size[0], size[1])]

    # Find XML files in those directories that match the category
    finalfiles = []
    for item in finaldirs:
        finalfiles += [(item, filename) for filename in sorted(listdir(item)) if category in filename and '.xml' in filename]
    return finalfiles