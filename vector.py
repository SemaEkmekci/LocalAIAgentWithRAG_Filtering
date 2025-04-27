# vector.py
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import json

# Örnek veri - gerçek uygulamada binlerce kaydınızı yükleyeceksiniz
cities = [
    {"id": 1, "city": "Adana"},
    {"id": 2, "city": "Adıyaman"},
    {"id": 3, "city": "Afyon"},
    {"id": 4, "city": "Ağrı"},
    {"id": 5, "city": "Amasya"},
    {"id": 6, "city": "Ankara"},
    {"id": 7, "city": "Antalya"},
    {"id": 8, "city": "Artvin"},
    {"id": 9, "city": "Aydın"},
    {"id": 42, "city": "Konya"}
]

tax_offices = [
    {"id": 1, "name": "5 Ocak Vergi Dairesi Müdürlüğü"},
    {"id": 2, "name": "Yüreğir Vergi Dairesi Müdürlüğü"},
    {"id": 3, "name": "Seyhan Vergi Dairesi Müdürlüğü"},
    {"id": 4, "name": "Ziyapaşa Vergi Dairesi Müdürlüğü"},
    {"id": 5, "name": "Çukurova Vergi Dairesi Müdürlüğü"},
    {"id": 6, "name": "Ceyhan Vergi Dairesi Müdürlüğü"},
    {"id": 7, "name": "Kozan Vergi Dairesi Müdürlüğü"},
    {"id": 8, "name": "Karataş Vergi Dairesi Müdürlüğü"},
    {"id": 20, "name": "Meram Vergi Dairesi Müdürlüğü"},
    {"id": 21, "name": "Selçuklu Vergi Dairesi Müdürlüğü"}
]

# Firma bilgilerini ekliyoruz (sadece id ve name alanları ile)
companies = [
    {"id": 101, "name": "AnadoluTech Bilişim A.Ş."},
    {"id": 102, "name": "Deniz İnşaat Ltd. Şti."},
    {"id": 103, "name": "Yıldız Gıda San. ve Tic. A.Ş."},
    {"id": 104, "name": "Akdeniz Otomotiv A.Ş."},
    {"id": 105, "name": "Yeşil Mobilya San. Ltd. Şti."},
    {"id": 106, "name": "Özgür Tekstil Ltd. Şti."},
    {"id": 107, "name": "Aydın Metal San. ve Tic. A.Ş."},
    {"id": 108, "name": "Güneş Enerji Sistemleri A.Ş."},
    {"id": 109, "name": "Yılmaz Tarım Ürünleri Ltd. Şti."},
    {"id": 110, "name": "Ankara Yazılım Teknolojileri A.Ş."},
    {"id": 111, "name": "Başkent Tıbbi Cihazlar San. A.Ş."},
    {"id": 112, "name": "Öztürk Danışmanlık Ltd. Şti."},
    {"id": 113, "name": "Kızılay Lojistik Hizmetleri A.Ş."},
    {"id": 114, "name": "Başkent Yapı Malzemeleri Ltd. Şti."},
    {"id": 115, "name": "Mavi Teknoloji Çözümleri A.Ş."},
    {"id": 116, "name": "Akdeniz Turizm İşletmeleri Ltd. Şti."},
    {"id": 117, "name": "Palmiye Otelcilik ve Turizm A.Ş."},
    {"id": 118, "name": "Başaran Gıda Ürünleri San. ve Tic. A.Ş."},
    {"id": 119, "name": "Artvin Orman Ürünleri A.Ş."},
    {"id": 120, "name": "Doğu Karadeniz Çay Sanayi Ltd. Şti."},
    {"id": 121, "name": "Aydın Tarım Makineleri A.Ş."},
    {"id": 122, "name": "Ege Zeytin ve Zeytinyağı İşletmeleri Ltd. Şti."},
    {"id": 123, "name": "Konya Tahıl Ürünleri San. ve Tic. A.Ş."},
    {"id": 124, "name": "Bozkır Hayvancılık Ltd. Şti."},
    {"id": 125, "name": "Meram Süt Ürünleri A.Ş."},
    {"id": 126, "name": "Selçuklu Makine Sanayi Ltd. Şti."},
    {"id": 127, "name": "Konya Döküm Sanayi A.Ş."},
    {"id": 128, "name": "Türkiye Petrolleri Anonim Ortaklığı"},
    {"id": 129, "name": "Devlet Su İşleri Genel Müdürlüğü"},
    {"id": 130, "name": "Türkiye İş Bankası A.Ş."}
]

# Embeddings oluşturma
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Şehir vektör deposu oluşturma
city_db_location = "./chroma_city_db"
add_city_documents = not os.path.exists(city_db_location)

if add_city_documents:
    city_documents = []
    city_ids = []
    
    for city in cities:
        document = Document(
            page_content=f"Şehir: {city['city']}",
            metadata={"id": city["id"], "type": "city"},
        )
        city_ids.append(f"city_{city['id']}")
        city_documents.append(document)
        
    city_vector_store = Chroma(
        collection_name="cities",
        persist_directory=city_db_location,
        embedding_function=embeddings
    )
    
    city_vector_store.add_documents(documents=city_documents, ids=city_ids)
else:
    city_vector_store = Chroma(
        collection_name="cities",
        persist_directory=city_db_location,
        embedding_function=embeddings
    )

# Vergi dairesi vektör deposu oluşturma
tax_db_location = "./chroma_tax_db"
add_tax_documents = not os.path.exists(tax_db_location)

if add_tax_documents:
    tax_documents = []
    tax_ids = []
    
    for office in tax_offices:
        document = Document(
            page_content=f"Vergi Dairesi: {office['name']}",
            metadata={"id": office["id"], "type": "tax"},
        )
        tax_ids.append(f"tax_{office['id']}")
        tax_documents.append(document)
        
    tax_vector_store = Chroma(
        collection_name="tax_offices",
        persist_directory=tax_db_location,
        embedding_function=embeddings
    )
    
    tax_vector_store.add_documents(documents=tax_documents, ids=tax_ids)
else:
    tax_vector_store = Chroma(
        collection_name="tax_offices",
        persist_directory=tax_db_location,
        embedding_function=embeddings
    )

# Şirket vektör deposu oluşturma
company_db_location = "./chroma_company_db"
add_company_documents = not os.path.exists(company_db_location)

if add_company_documents:
    company_documents = []
    company_ids = []
    
    for company in companies:
    
        document = Document(
            page_content=f"Firma: {company['name']}",
            metadata={"id": company["id"], "type": "company"},
        )
        company_ids.append(f"company_{company['id']}")
        company_documents.append(document)
        
    company_vector_store = Chroma(
        collection_name="companies",
        persist_directory=company_db_location,
        embedding_function=embeddings
    )
    
    company_vector_store.add_documents(documents=company_documents, ids=company_ids)
else:
    company_vector_store = Chroma(
        collection_name="companies",
        persist_directory=company_db_location,
        embedding_function=embeddings
    )

# Retriever'ları oluşturma
city_retriever = city_vector_store.as_retriever(search_kwargs={"k": 3})
tax_retriever = tax_vector_store.as_retriever(search_kwargs={"k": 3})
company_retriever = company_vector_store.as_retriever(search_kwargs={"k": 5})

def search_data(query):
    # Her üç koleksiyonu da ara
    cities_results = city_retriever.invoke(query)
    tax_results = tax_retriever.invoke(query)
    company_results = company_retriever.invoke(query)
    
    # ID'leri çıkar ve integer olduklarından emin ol
    city_ids = [int(doc.metadata["id"]) for doc in cities_results]
    tax_ids = [int(doc.metadata["id"]) for doc in tax_results]
    company_ids = [int(doc.metadata["id"]) for doc in company_results]
    
    # Sonuçları döndür
    return {
        "cities": city_ids,
        "tax": tax_ids,
        "companies": company_ids
    }

# Doğrudan arama fonksiyonu - regex ile şehir ve vergi dairesi isimleri aranır
def direct_search(query):
    """
    Metin içinde doğrudan şehir, vergi dairesi ve firma isimlerini arar
    ve sadece açıkça belirtilen öğeleri döndürür
    """
    city_ids = []
    tax_ids = []
    company_ids = []
    
    # Küçük harfe çevir
    query_lower = query.lower()
    
    # Şehirleri kontrol et
    for city in cities:
        if city["city"].lower() in query_lower:
            city_ids.append(city["id"])
    
    # Vergi dairelerini kontrol et
    for office in tax_offices:
        # Vergi dairesi ismi büyük ihtimalle birden fazla kelimeden oluşuyor
        # Bu nedenle en azından iki önemli kelimeyi içerip içermediğini kontrol et
        office_words = [word.lower() for word in office["name"].split() if len(word) > 3]
        found_words = [word for word in office_words if word in query_lower]
        
        # Eğer vergi dairesi adının önemli kelimelerinden en az ikisi sorguda varsa
        if len(found_words) >= 2 or any(part + " vergi" in query_lower for part in office_words):
            tax_ids.append(office["id"])
            # Artık otomatik olarak şehir ID'si ekleme - sadece açıkça belirtilenleri ekle
    
    # Firmaları kontrol et
    for company in companies:
        # Firma ismi kontrolü
        company_name_lower = company["name"].lower()
        
        # Firma isminin bir kısmı veya tamamı sorgu içinde geçiyorsa
        if (company_name_lower in query_lower or 
            any(len(part) > 3 and part.lower() in query_lower for part in company["name"].split())):
            company_ids.append(company["id"])
            # Artık otomatik olarak şehir/vergi dairesi ID'si ekleme
    
    return {
        "cities": city_ids,
        "tax": tax_ids,
        "companies": company_ids
    }