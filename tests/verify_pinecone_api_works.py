import pinecone


def main():
    pinecone.init(api_key="YOUR_API_KEY", environment="YOUR_ENVIRONMENT")
    pinecone.list_indexes()

