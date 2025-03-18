.PHONY: default docker
default::
	pip install -e .
docker:
	docker compose -f docker_compose.yaml run --build hierarchical_3d_gaussians


DOCKER_COMPOSE_PATH=docker_compose.yaml

build:
	docker compose -f $(DOCKER_COMPOSE_PATH) build hierarchical_3d_gaussians	

run:
	docker compose -f $(DOCKER_COMPOSE_PATH) run hierarchical_3d_gaussians