from gpt_app_learn.utils.page import PDFQAContent


def main():
    QDRANT_PATH = "./local_qdrant"
    COLLECTION_NAME = "my_collection_2"

    page_content = PDFQAContent(
        qdrant_root=QDRANT_PATH, collection_name=COLLECTION_NAME
    )
    page_content.init_page(
        page_title="Ask My PDF(s)",
        page_icon="🤗",
        header_title="Ask My PDF(s)",
        sidebar_title="Nav",
    )
    page_content.init_costs()
    page_content.switch_page()
    page_content.print_costs()


if __name__ == "__main__":
    main()
