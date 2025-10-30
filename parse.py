import os
import sys
from parsestudio.parse import PDFParser

def parse_pdfs_to_txt(input_folder, output_txt):
    parser = PDFParser(parser="pymupdf")

    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".pdf")]
    pdf_files.sort()

    with open(output_txt, "w", encoding="utf-8") as outfile:
        for pdf_file in pdf_files:
            pdf_path = os.path.join(input_folder, pdf_file)
            print(f"Processing: {pdf_path}")

            try:
                outputs = parser.run([pdf_path], modalities=["text"])
                text_data = outputs[0].text

                # Convert TextElement(s) to string
                if isinstance(text_data, list):
                    text_content = "\n".join([t.text if hasattr(t, "text") else str(t) for t in text_data])
                else:
                    text_content = text_data.text if hasattr(text_data, "text") else str(text_data)

                outfile.write(f"===== {pdf_file} =====\n")
                outfile.write(text_content + "\n\n")

            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")

    print(f"✅ All text saved to {output_txt}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python parse_all_pdfs.py <input_folder> <output_txt>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_txt = sys.argv[2]

    parse_pdfs_to_txt(input_folder, output_txt)
