import requests
import json

class SimpleNLPClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def load_texts_from_file(self, filepath="data/texts.txt"):
        """Загрузка текстов из файла"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Разделяем по двойным переносам строк (пустым строкам)
                texts = [text.strip() for text in content.split('\n\n') if text.strip()]
                if not texts:  # Если файл пустой
                    print(f"Файл {filepath} пустой. Использую пример текстов.")
                    return self.get_sample_texts()
                return texts
        except FileNotFoundError:
            print(f"Файл {filepath} не найден. Использую пример текстов.")
            return self.get_sample_texts()
        except Exception as e:
            print(f"Ошибка при чтении файла: {e}. Использую пример текстов.")
            return self.get_sample_texts()
    
    def get_sample_texts(self):
        """Возвращает пример текстов если файла нет"""
        return [
            "Natural language processing helps computers understand human language.",
            "Machine learning algorithms learn from data.",
            "Deep learning uses neural networks with many layers.",
            "Python is a popular programming language for AI.",
            "FastAPI makes it easy to build web APIs."
        ]
    
    def test_all_endpoints(self):
        """Тестирует все эндпоинты"""
        print("=" * 50)
        print("Тестирование NLP микросервиса")
        print("=" * 50)
        
        # Загружаем тексты
        texts = self.load_texts_from_file()
        if not texts:  # Проверяем, что тексты не пустые
            print("❌ Не удалось загрузить тексты для тестирования")
            return False
            
        print(f"✅ Загружено {len(texts)} текстов для обработки")
        
        # Тестируем TF-IDF
        print("\n1. Тестируем TF-IDF:")
        try:
            response = requests.post(f"{self.base_url}/tf-idf", json={
                "texts": texts[:3],  # первые 3 текста
                "max_features": 20
            }, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Успешно! Матрица размером: {data['shape']}")
            else:
                print(f"   ❌ Ошибка {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Ошибка подключения: {e}")
        
        # Тестируем Bag of Words
        print("\n2. Тестируем Bag of Words:")
        try:
            response = requests.post(f"{self.base_url}/bag-of-words", json={
                "texts": texts[:3],
                "max_features": 20
            }, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Успешно! Матрица размером: {data['shape']}")
            else:
                print(f"   ❌ Ошибка {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Ошибка подключения: {e}")
        
        # Тестируем NLTK операции
        print("\n3. Тестируем NLTK операции:")
        test_text = texts[0]
        
        # Токенизация
        print("   а) Токенизация:")
        try:
            response = requests.post(f"{self.base_url}/text_nltk/tokenize", 
                                   json={"text": test_text},
                                   timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"      ✅ {len(data['tokens'])} токенов: {data['tokens'][:5]}...")
            else:
                print(f"      ❌ Ошибка {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"      ❌ Ошибка подключения: {e}")
        
        # POS тегинг
        print("   б) POS тегинг:")
        try:
            response = requests.post(f"{self.base_url}/text_nltk/pos_tag",
                                   json={"text": test_text},
                                   timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"      ✅ {len(data['pos_tags'])} тегов: {data['pos_tags'][:5]}...")
            else:
                print(f"      ❌ Ошибка {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"      ❌ Ошибка подключения: {e}")
        
        print("\n" + "=" * 50)
        print("Для полного тестирования используйте:")
        print(f"  - Swagger UI: {self.base_url}/docs")
        print(f"  - ReDoc: {self.base_url}/redoc")
        print("=" * 50)
        return True

def quick_test():
    """Быстрая проверка работы сервера"""
    client = SimpleNLPClient()
    
    try:
        # Проверяем доступность сервера
        print("Проверка подключения к серверу...")
        response = requests.get(f"{client.base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Сервер работает!")
            print(f"Сообщение: {response.json()['message']}")
            
            # Запускаем полный тест
            client.test_all_endpoints()
        else:
            print(f"❌ Сервер вернул код: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Не могу подключиться к серверу")
        print("\nУбедитесь, что:")
        print("1. Сервер запущен на порту 8000")
        print("2. Запустите сервер командой:")
        print("   cd server && python main.py")
    except requests.exceptions.Timeout:
        print("❌ Таймаут подключения к серверу")
    except Exception as e:
        print(f"❌ Неизвестная ошибка: {e}")

if __name__ == "__main__":
    quick_test()