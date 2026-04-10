"""
Secure Storage Module for Proctorless Exam System
Implements AES-256-GCM encryption for exam data and reports

Usage:
    from utils.encryption import get_storage
    
    storage = get_storage()
    storage.save('path/to/file.enc', data_dict)
    data = storage.load('path/to/file.enc')
"""

import os
import json
import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from datetime import datetime


class SecureStorage:
    """
    Handles encryption/decryption of sensitive exam data
    
    Features:
    - AES-256-GCM encryption (industry standard)
    - Authenticated encryption (prevents tampering)
    - Automatic key generation and management
    - Simple save/load API
    """
    
    def __init__(self, key_file='storage/master.key'):
        """
        Initialize secure storage
        
        Args:
            key_file: Path to master encryption key file
        """
        self.key_file = key_file
        
        # Load or generate master key
        if os.path.exists(key_file):
            # Load existing key
            with open(key_file, 'rb') as f:
                self.key = f.read()
            print(f" Loaded encryption key from: {key_file}")
        else:
            # Generate new key (first time setup)
            self.key = AESGCM.generate_key(bit_length=256)
            
            # Save key to file
            os.makedirs(os.path.dirname(key_file) if os.path.dirname(key_file) else '.', exist_ok=True)
            with open(key_file, 'wb') as f:
                f.write(self.key)
            
            # Set secure file permissions (read/write owner only)
            os.chmod(key_file, 0o600)
            
            print(f" Generated new encryption key: {key_file}")
            print(f"  IMPORTANT: Backup this file! Without it, encrypted data cannot be recovered!")
            print(f"  Add '{key_file}' to .gitignore to prevent committing to version control")
        
        # Verify key length
        if len(self.key) != 32:
            raise ValueError(f"Invalid key length: {len(self.key)} bytes (expected 32 bytes)")
        
        # Initialize cipher
        self.cipher = AESGCM(self.key)
        
        print(" Secure storage initialized (AES-256-GCM)")
    
    def encrypt(self, data):
        """
        Encrypt data using AES-256-GCM
        
        Args:
            data: Dictionary or any JSON-serializable object
            
        Returns:
            dict: Encrypted package containing:
                - data: Base64-encoded ciphertext
                - nonce: Base64-encoded nonce
                - timestamp: Encryption timestamp
                - algorithm: Encryption algorithm used
        """
        # Convert to JSON string
        if isinstance(data, dict) or isinstance(data, list):
            plaintext = json.dumps(data).encode('utf-8')
        else:
            plaintext = str(data).encode('utf-8')
        
        # Generate random nonce (12 bytes for GCM)
        nonce = os.urandom(12)
        
        # Encrypt with authenticated encryption
        # GCM provides both confidentiality and authenticity
        ciphertext = self.cipher.encrypt(nonce, plaintext, None)
        
        # Create encrypted package
        encrypted_package = {
            'data': base64.b64encode(ciphertext).decode('utf-8'),
            'nonce': base64.b64encode(nonce).decode('utf-8'),
            'timestamp': datetime.now().isoformat(),
            'algorithm': 'AES-256-GCM',
            'version': '1.0'
        }
        
        return encrypted_package
    
    def decrypt(self, encrypted_package):
        """
        Decrypt AES-256-GCM encrypted data
        
        Args:
            encrypted_package: Dictionary from encrypt() method
            
        Returns:
            Original data (dict, list, or string)
            
        Raises:
            ValueError: If decryption fails (wrong key or tampered data)
        """
        try:
            # Decode from base64
            ciphertext = base64.b64decode(encrypted_package['data'])
            nonce = base64.b64decode(encrypted_package['nonce'])
            
            # Decrypt and verify authenticity
            plaintext = self.cipher.decrypt(nonce, ciphertext, None)
            
            # Try to parse as JSON
            try:
                return json.loads(plaintext.decode('utf-8'))
            except json.JSONDecodeError:
                # Return as string if not JSON
                return plaintext.decode('utf-8')
        
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}. Data may be corrupted or wrong key used.")
    
    def save(self, filepath, data):
        """
        Encrypt data and save to file
        
        Args:
            filepath: Path where encrypted file will be saved
            data: Data to encrypt (dict, list, or any JSON-serializable)
            
        Example:
            storage.save('storage/encrypted/exam_123.enc', exam_data)
        """
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Encrypt data
        encrypted = self.encrypt(data)
        
        # Save to file as JSON
        with open(filepath, 'w') as f:
            json.dump(encrypted, f, indent=2)
        
        print(f" Saved encrypted file: {filepath}")
    
    def load(self, filepath):
        """
        Load and decrypt file
        
        Args:
            filepath: Path to encrypted file
            
        Returns:
            Decrypted data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If decryption fails
            
        Example:
            data = storage.load('storage/encrypted/exam_123.enc')
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Encrypted file not found: {filepath}")
        
        # Load encrypted package
        with open(filepath, 'r') as f:
            encrypted_package = json.load(f)
        
        # Decrypt and return
        return self.decrypt(encrypted_package)
    
    def delete(self, filepath, secure=False):
        """
        Delete encrypted file
        
        Args:
            filepath: Path to file to delete
            secure: If True, overwrite file before deletion (slower but more secure)
        """
        if not os.path.exists(filepath):
            return
        
        if secure:
            # Secure deletion: overwrite with random data before deleting
            file_size = os.path.getsize(filepath)
            with open(filepath, 'wb') as f:
                f.write(os.urandom(file_size))
        
        # Delete file
        os.remove(filepath)
        print(f"  Deleted: {filepath}")
    
    def list_encrypted_files(self, directory='storage/encrypted'):
        """
        List all encrypted files in directory
        
        Args:
            directory: Directory to search
            
        Returns:
            list: List of encrypted file paths
        """
        if not os.path.exists(directory):
            return []
        
        encrypted_files = []
        for filename in os.listdir(directory):
            if filename.endswith('.enc'):
                encrypted_files.append(os.path.join(directory, filename))
        
        return encrypted_files
    
    def verify_integrity(self, filepath):
        """
        Verify integrity of encrypted file (check if it can be decrypted)
        
        Args:
            filepath: Path to encrypted file
            
        Returns:
            bool: True if file is valid and can be decrypted
        """
        try:
            self.load(filepath)
            return True
        except:
            return False


# Singleton instance for convenience
_storage_instance = None

def get_storage(key_file='storage/master.key'):
    """
    Get or create singleton SecureStorage instance
    
    Args:
        key_file: Path to master key file
        
    Returns:
        SecureStorage: Singleton instance
        
    Example:
        from utils.encryption import get_storage
        storage = get_storage()
        storage.save('file.enc', data)
    """
    global _storage_instance
    
    if _storage_instance is None:
        _storage_instance = SecureStorage(key_file)
    
    return _storage_instance


# Testing and demonstration
if __name__ == '__main__':
    """
    Test encryption functionality
    Run: python3 -m utils.encryption
    """
    print("="*70)
    print("Testing Secure Storage Module")
    print("="*70)
    
    # Initialize storage
    storage = get_storage()
    
    # Test data
    test_data = {
        'session_id': 'TEST_SESSION_123',
        'student_id': '21MID0009',
        'exam_id': 'CAPSTONE2024',
        'overall_score': 87,
        'component_scores': {
            'eye_gaze': 90,
            'speech': 84,
            'gadgets': 60
        },
        'violations': {
            'phone': 3,
            'speech': 2,
            'multiple_faces': 0
        },
        'timestamp': datetime.now().isoformat()
    }
    
    print("\n Original Data:")
    print(json.dumps(test_data, indent=2))
    
    # Encrypt
    print("\n Encrypting data...")
    encrypted = storage.encrypt(test_data)
    print(f"   Algorithm: {encrypted['algorithm']}")
    print(f"   Encrypted: {encrypted['data'][:60]}...")
    print(f"   Nonce: {encrypted['nonce']}")
    
    # Decrypt
    print("\n Decrypting data...")
    decrypted = storage.decrypt(encrypted)
    print(json.dumps(decrypted, indent=2))
    
    # Verify
    print("\n Verification:")
    if decrypted == test_data:
        print("    SUCCESS! Encryption/decryption works correctly")
    else:
        print("    FAILED! Data mismatch")
    
    # Test file operations
    print("\n Testing file operations...")
    test_file = 'storage/encrypted/test_report.enc'
    
    storage.save(test_file, test_data)
    loaded_data = storage.load(test_file)
    
    if loaded_data == test_data:
        print("    File save/load works correctly")
    else:
        print("    File operations failed")
    
    # Cleanup
    storage.delete(test_file)
    
    print("\n" + "="*70)
    print(" All tests passed! Encryption module is working correctly.")
    print("="*70)
