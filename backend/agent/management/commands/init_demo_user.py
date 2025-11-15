from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model

User = get_user_model()

class Command(BaseCommand):
    help = 'Инициализация демо-пользователя для разработки'

    def handle(self, *args, **options):
        username = 'demo_user'
        email = 'demo@raifrag.local'
        
        user, created = User.objects.get_or_create(
            username=username,
            defaults={
                'email': email,
                'first_name': 'Демо',
                'last_name': 'Пользователь',
                'is_active': True,
                'is_staff': False,
                'is_superuser': False
            }
        )
        
        if created:
            # Устанавливаем пароль (опционально)
            user.set_password('demo123456')
            user.save()
            self.stdout.write(
                self.style.SUCCESS(f'✅ Создан демо-пользователь: {username}')
            )
        else:
            self.stdout.write(
                self.style.WARNING(f'⚠️ Пользователь {username} уже существует')
            )
        
        self.stdout.write('━' * 60)
        self.stdout.write(f'Username: {user.username}')
        self.stdout.write(f'Email: {user.email}')
        self.stdout.write(f'ID: {user.id}')
        self.stdout.write('━' * 60)