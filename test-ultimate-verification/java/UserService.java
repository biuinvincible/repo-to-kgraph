/**
 * Java service class with inheritance and interfaces.
 * Tests Java parsing capabilities.
 */
package com.example.service;

import java.util.List;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;

public interface Repository<T, ID> {
    Optional<T> findById(ID id);
    List<T> findAll();
    T save(T entity);
    void deleteById(ID id);
}

public abstract class BaseService<T, ID> implements Repository<T, ID> {
    protected final String serviceName;

    public BaseService(String serviceName) {
        this.serviceName = serviceName;
    }

    protected abstract void validateEntity(T entity);

    public abstract CompletableFuture<T> saveAsync(T entity);
}

@Service
@Transactional
public class UserService extends BaseService<User, Long> {
    private final UserRepository userRepository;
    private final EmailService emailService;

    public UserService(UserRepository userRepository, EmailService emailService) {
        super("UserService");
        this.userRepository = userRepository;
        this.emailService = emailService;
    }

    @Override
    public Optional<User> findById(Long id) {
        return userRepository.findById(id);
    }

    @Override
    public List<User> findAll() {
        return userRepository.findAll();
    }

    @Override
    public User save(User user) {
        validateEntity(user);
        User savedUser = userRepository.save(user);
        emailService.sendWelcomeEmail(savedUser.getEmail());
        return savedUser;
    }

    @Override
    public void deleteById(Long id) {
        userRepository.deleteById(id);
    }

    @Override
    protected void validateEntity(User user) {
        if (user.getEmail() == null || user.getEmail().isEmpty()) {
            throw new IllegalArgumentException("Email cannot be null or empty");
        }
    }

    @Override
    public CompletableFuture<User> saveAsync(User user) {
        return CompletableFuture.supplyAsync(() -> save(user));
    }

    public List<User> findActiveUsers() {
        return userRepository.findByStatus(UserStatus.ACTIVE);
    }
}