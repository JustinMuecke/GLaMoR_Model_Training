package org.example;

import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.formats.RDFXMLDocumentFormat;
import org.semanticweb.owlapi.model.IRI;
import org.semanticweb.owlapi.model.OWLOntology;
import org.semanticweb.owlapi.model.OWLOntologyManager;

import java.io.File;

public class Main {
    public static void main(String[] args) {
        try {
            // Create ontology manager
            OWLOntologyManager manager = OWLManager.createOWLOntologyManager();

            // Input and output directories
            File inputDirectory = new File("../data/ont_modules_inconsistent");
            File outputDirectory = new File("../data/ont_modules_inconsistent_rdf");

            // Ensure output directory exists
            if (!outputDirectory.exists()) {
                outputDirectory.mkdirs();
            }

            // Process each file in the input directory
            File[] ontologyFiles = inputDirectory.listFiles((dir, name) -> name.endsWith(".owl"));

            if (ontologyFiles == null || ontologyFiles.length == 0) {
                System.out.println("No ontology files found in the input directory.");
                return;
            }

            for (File inputFile : ontologyFiles) {
                try {
                    System.out.println("Processing: " + inputFile.getName());

                    // Load ontology
                    OWLOntology ontology = manager.loadOntologyFromOntologyDocument(inputFile);

                    // Define output file path
                    File outputFile = new File(outputDirectory, inputFile.getName());

                    // Save ontology in RDF/XML format
                    RDFXMLDocumentFormat format = new RDFXMLDocumentFormat();
                    manager.saveOntology(ontology, format, IRI.create(outputFile.toURI()));

                    System.out.println("Converted and saved: " + outputFile.getName());
                } catch (Exception e) {
                    System.err.println("Error processing file: " + inputFile.getName());
                    e.printStackTrace();
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}