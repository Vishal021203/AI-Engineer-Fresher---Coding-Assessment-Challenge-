import pandas as pd
import re
import heapq
from datetime import datetime
from collections import defaultdict
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import textwrap

# Download NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class CLISupportSystem:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.priority_keywords = ['urgent', 'critical', 'immediately', 'emergency', 
                                'cannot access', 'blocked', 'down', 'broken', 'not working']
        
        # Knowledge base for response generation
        self.knowledge_base = {
            'account_verification': "Account verification emails are sent immediately after registration. If not received, check spam folder or request a new verification link.",
            'login_issues': "Common login issues include incorrect passwords, browser cache problems, or account lockouts. Try resetting password or clearing browser cache.",
            'billing_errors': "Billing issues are handled by our finance team. Refunds typically process within 5-7 business days after approval.",
            'api_integration': "We support REST API integration with comprehensive documentation available at api.docs.example.com.",
            'subscription': "We offer Basic ($29/mo), Pro ($79/mo), and Enterprise ($199/mo) plans. All include 24/7 support and API access.",
            'password_reset': "Password reset links expire after 1 hour. If the link doesn't work, request a new one from the login page."
        }
        
        self.emails = []
        self.processed_emails = []
        self.priority_queue = []
        self.email_counter = 0
        
    def load_emails(self, csv_path):
        """Load emails from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                email = {
                    'sender': row['sender'],
                    'subject': row['subject'],
                    'body': row['body'],
                    'sent_date': datetime.strptime(row['sent_date'], '%Y-%m-%d %H:%M:%S'),
                    'id': f"email_{len(self.emails) + 1}"
                }
                self.emails.append(email)
            print(f"✓ Loaded {len(self.emails)} emails from {csv_path}")
            return True
        except Exception as e:
            print(f"✗ Error loading emails: {e}")
            return False
    
    def filter_emails(self):
        """Filter emails based on support-related keywords"""
        support_keywords = ['support', 'query', 'request', 'help', 'assist', 'issue', 'problem']
        
        filtered_emails = []
        for email in self.emails:
            subject_lower = email['subject'].lower()
            body_lower = email['body'].lower()
            
            if any(keyword in subject_lower for keyword in support_keywords) or \
               any(keyword in body_lower for keyword in support_keywords):
                filtered_emails.append(email)
        
        return filtered_emails
    
    def analyze_sentiment(self, text):
        """Perform sentiment analysis on email text"""
        scores = self.sia.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            return 'Positive', compound
        elif compound <= -0.05:
            return 'Negative', compound
        else:
            return 'Neutral', compound
    
    def determine_priority(self, email):
        """Determine priority based on keywords and sentiment"""
        text = f"{email['subject']} {email['body']}".lower()
        
        # Check for priority keywords
        urgency_score = sum(1 for keyword in self.priority_keywords if keyword in text)
        
        # Check sentiment
        sentiment, score = self.analyze_sentiment(text)
        
        # Negative sentiment increases priority
        if sentiment == 'Negative':
            urgency_score += 2
        elif sentiment == 'Positive':
            urgency_score -= 1
        
        # Time sensitivity (recent emails get higher priority)
        hours_old = (datetime.now() - email['sent_date']).total_seconds() / 3600
        if hours_old < 1:
            urgency_score += 3
        elif hours_old < 4:
            urgency_score += 2
        elif hours_old < 12:
            urgency_score += 1
        
        return 'Urgent' if urgency_score >= 3 else 'Not urgent', urgency_score
    
    def extract_information(self, email):
        """Extract key information from email"""
        text = f"{email['subject']} {email['body']}"
        
        # Extract phone numbers
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phones = re.findall(phone_pattern, text)
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        
        # Extract potential requirements
        requirements = []
        requirement_keywords = ['need', 'want', 'require', 'looking for', 'help with']
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in requirement_keywords):
                requirements.append(sentence.strip())
        
        return {
            'phone_numbers': phones,
            'alternate_emails': [email for email in emails if email != email['sender']],
            'requirements': requirements[:3],  # Top 3 requirements
            'sentiment': self.analyze_sentiment(text)[0]
        }
    
    def categorize_email(self, email):
        """Categorize email based on content"""
        text = f"{email['subject']} {email['body']}".lower()
        
        categories = {
            'account_issues': ['account', 'verify', 'verification', 'login', 'password'],
            'billing': ['billing', 'payment', 'charge', 'refund', 'invoice'],
            'technical': ['api', 'integration', 'technical', 'server', 'system'],
            'general': ['question', 'query', 'information', 'help']
        }
        
        scores = {category: 0 for category in categories}
        
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in text:
                    scores[category] += 1
        
        # Return category with highest score
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def generate_response(self, email, extracted_info):
        """Generate context-aware response using rule-based approach"""
        category = self.categorize_email(email)
        sentiment = extracted_info['sentiment']
        
        # Customize response based on sentiment
        if sentiment == 'Negative':
            greeting = "Thank you for bringing this to our attention. We sincerely apologize for the inconvenience you've experienced."
            empathy = "We understand your frustration and are committed to resolving this issue promptly."
        elif sentiment == 'Positive':
            greeting = "Thank you for reaching out to us! We appreciate your feedback."
            empathy = "We're glad to hear from you and are happy to assist."
        else:
            greeting = "Thank you for contacting our support team."
            empathy = "We're here to help and will address your inquiry promptly."
        
        # Add knowledge base information
        knowledge = self.knowledge_base.get(category, "Our team is looking into your request and will provide assistance shortly.")
        
        # Build response
        response = f"""
{greeting}

{empathy}

Regarding your {category.replace('_', ' ')}: {knowledge}

Please don't hesitate to reach out if you need any further assistance.

Best regards,
Support Team
"""
        
        return textwrap.dedent(response).strip()
    
    def process_emails(self):
        """Process all filtered emails and add to priority queue"""
        filtered_emails = self.filter_emails()
        
        for email in filtered_emails:
            # Analyze and categorize
            sentiment, score = self.analyze_sentiment(f"{email['subject']} {email['body']}")
            priority, urgency_score = self.determine_priority(email)
            category = self.categorize_email(email)
            extracted_info = self.extract_information(email)
            response = self.generate_response(email, extracted_info)
            
            # Create processed email object
            processed_email = {
                **email,
                'sentiment': sentiment,
                'priority': priority,
                'urgency_score': urgency_score,
                'category': category,
                'extracted_info': extracted_info,
                'generated_response': response,
                'status': 'Pending',
                'processing_id': self.email_counter  # Add unique ID for heapq comparison
            }
            
            # Add to priority queue with unique ID to avoid comparison issues
            heapq.heappush(self.priority_queue, (-urgency_score, self.email_counter, processed_email))
            self.processed_emails.append(processed_email)
            self.email_counter += 1
        
        print(f"✓ Processed {len(filtered_emails)} support emails")
        return len(filtered_emails)
    
    def display_analytics(self):
        """Display analytics and statistics"""
        if not self.processed_emails:
            print("No emails processed yet.")
            return
        
        # Calculate statistics
        total_emails = len(self.processed_emails)
        resolved_emails = sum(1 for email in self.processed_emails if email.get('status') == 'Resolved')
        pending_emails = total_emails - resolved_emails
        
        # Count by category
        sentiment_counts = defaultdict(int)
        priority_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for email in self.processed_emails:
            sentiment_counts[email['sentiment']] += 1
            priority_counts[email['priority']] += 1
            category_counts[email['category']] += 1
        
        print("\n" + "="*60)
        print("ANALYTICS & STATISTICS")
        print("="*60)
        print(f"Total support emails: {total_emails}")
        print(f"Resolved: {resolved_emails}")
        print(f"Pending: {pending_emails}")
        
        print("\nSENTIMENT DISTRIBUTION:")
        for sentiment, count in sentiment_counts.items():
            print(f"  {sentiment}: {count}")
            
        print("\nPRIORITY DISTRIBUTION:")
        for priority, count in priority_counts.items():
            print(f"  {priority}: {count}")
            
        print("\nCATEGORY DISTRIBUTION:")
        for category, count in category_counts.items():
            print(f"  {category.replace('_', ' ').title()}: {count}")
        print("="*60)
    
    def display_emails(self, priority_filter=None):
        """Display processed emails with filtering options"""
        if not self.processed_emails:
            print("No emails to display.")
            return
        
        # Get emails in priority order (ignore the score and ID from heapq)
        priority_emails = [email for _, _, email in sorted(self.priority_queue, key=lambda x: x[0])]
        
        # Apply filter if specified
        if priority_filter:
            filtered_emails = [email for email in priority_emails if email['priority'] == priority_filter]
            if not filtered_emails:
                print(f"No {priority_filter} emails found.")
                return
            display_emails = filtered_emails
        else:
            display_emails = priority_emails
        
        print(f"\n{'='*100}")
        print(f"SUPPORT EMAILS{' (' + priority_filter + ')' if priority_filter else ''}")
        print(f"{'='*100}")
        
        for i, email in enumerate(display_emails, 1):
            print(f"\n{i}. [{email['priority']}] {email['subject']}")
            print(f"   From: {email['sender']}")
            print(f"   Date: {email['sent_date'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Category: {email['category'].replace('_', ' ').title()}")
            print(f"   Sentiment: {email['sentiment']}")
            print(f"   Status: {email.get('status', 'Pending')}")
            
            # Show extracted information
            ext = email['extracted_info']
            if ext['phone_numbers']:
                print(f"   Phone Numbers: {', '.join(ext['phone_numbers'])}")
            if ext['alternate_emails']:
                print(f"   Alternate Emails: {', '.join(ext['alternate_emails'])}")
            if ext['requirements']:
                print(f"   Key Requirements:")
                for req in ext['requirements']:
                    print(f"     - {req}")
            
            print(f"\n   Body: {textwrap.shorten(email['body'], width=80, placeholder='...')}")
            
            # Show response preview
            response_preview = textwrap.shorten(email['generated_response'], width=80, placeholder='...')
            print(f"   AI Response: {response_preview}")
            
            print(f"{'-'*100}")
    
    def show_email_detail(self, index):
        """Show detailed view of a specific email"""
        if index < 1 or index > len(self.processed_emails):
            print("Invalid email index.")
            return
        
        # Get emails in priority order
        priority_emails = [email for _, _, email in sorted(self.priority_queue, key=lambda x: x[0])]
        email = priority_emails[index-1]
        
        print(f"\n{'='*100}")
        print(f"EMAIL DETAIL: {email['subject']}")
        print(f"{'='*100}")
        print(f"From: {email['sender']}")
        print(f"Date: {email['sent_date'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Category: {email['category'].replace('_', ' ').title()}")
        print(f"Sentiment: {email['sentiment']}")
        print(f"Priority: {email['priority']}")
        print(f"Status: {email.get('status', 'Pending')}")
        
        # Show extracted information
        ext = email['extracted_info']
        print(f"\nEXTRACTED INFORMATION:")
        if ext['phone_numbers']:
            print(f"  Phone Numbers: {', '.join(ext['phone_numbers'])}")
        if ext['alternate_emails']:
            print(f"  Alternate Emails: {', '.join(ext['alternate_emails'])}")
        if ext['requirements']:
            print(f"  Key Requirements:")
            for req in ext['requirements']:
                print(f"    - {req}")
        
        print(f"\nFULL BODY:")
        print(textwrap.fill(email['body'], width=80))
        
        print(f"\nAI-GENERATED RESPONSE:")
        print(textwrap.fill(email['generated_response'], width=80))
        print(f"{'='*100}")
        
        # Offer options
        print("\nOptions:")
        print("1. Mark as resolved")
        print("2. Edit response")
        print("3. Back to list")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice == "1":
            email['status'] = 'Resolved'
            print("✓ Email marked as resolved.")
        elif choice == "2":
            new_response = input("Enter your edited response (press Enter to keep current): ").strip()
            if new_response:
                email['generated_response'] = new_response
                print("✓ Response updated.")
    
    def run_cli(self):
        """Run the command-line interface"""
        print("🤖 AI-Powered Support Email System")
        print("="*50)
        
        # Load emails
        csv_file = "68b1acd44f393_Sample_Support_Emails_Dataset.csv"
        if not self.load_emails(csv_file):
            return
        
        # Process emails
        processed_count = self.process_emails()
        if processed_count == 0:
            print("No support emails found to process.")
            return
        
        # Main menu loop
        while True:
            print("\nMAIN MENU")
            print("1. View all emails")
            print("2. View urgent emails only")
            print("3. View analytics")
            print("4. View email detail")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                self.display_emails()
            elif choice == "2":
                self.display_emails("Urgent")
            elif choice == "3":
                self.display_analytics()
            elif choice == "4":
                try:
                    index = int(input("Enter email number to view details: "))
                    self.show_email_detail(index)
                except ValueError:
                    print("Please enter a valid number.")
            elif choice == "5":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")

# Run the CLI application
if __name__ == "__main__":
    system = CLISupportSystem()
    system.run_cli()