module github.com/dobbo-ca/cadmus/oracle

go 1.27.1

require github.com/dobbo-ca/cadmus v0.0.0

// The oracle's module path is below cadmus's so that Go's internal-package rule lets it import
// cadmus/internal/...; the upstream snapshot is the module it resolves to.
replace github.com/dobbo-ca/cadmus => ../upstream
