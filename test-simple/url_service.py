"""
URL Service - Create and manage shortened URLs
"""

class URLService:
    def __init__(self):
        self.urls = {}

    def create_url(self, original_url, custom_code=None):
        """Create a new shortened URL with optional custom code."""
        if custom_code:
            short_code = custom_code
        else:
            short_code = self.generate_code()

        url_data = {
            'original_url': original_url,
            'short_code': short_code,
            'clicks': 0
        }

        self.urls[short_code] = url_data
        return url_data

    def generate_code(self):
        """Generate a random 6-character code."""
        import random
        import string
        return ''.join(random.choices(string.ascii_letters + string.digits, k=6))

    def get_analytics(self, short_code):
        """Get click statistics for a URL."""
        if short_code in self.urls:
            return {
                'clicks': self.urls[short_code]['clicks'],
                'original_url': self.urls[short_code]['original_url']
            }
        return None